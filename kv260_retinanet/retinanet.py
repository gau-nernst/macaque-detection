import argparse

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.retinanet import RetinaNetHead, RetinaNet
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from tqdm import tqdm

from pytorch_nndct.apis import torch_quantizer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# override forward behavior of RetinaNet head. remove reshape code
class Head(RetinaNetHead):
    def forward(self, x):
        outputs = []

        for features in x:
            logits = self.classification_head.conv(features)
            logits = self.classification_head.cls_logits(logits)        # N x A*K x H x W

            bboxes = self.regression_head.conv(features)
            bboxes = self.regression_head.bbox_reg(bboxes)              # N x A*4 x H x W

            outputs.append(torch.cat((logits, bboxes), dim=1))          # N x A*(K+4) x H x W

        return outputs


# override forward behavior of RetinaNet. remove post-processing codes
class Model(RetinaNet):
    def forward(self, x):
        features = list(self.backbone(x).values())
        return self.head(features)


def get_quantized_model(args):
    backbone = resnet_fpn_backbone(
        'resnet50', False, returned_layers=[2, 3, 4],
        # extra_blocks=LastLevelP6P7(256, 256),         # this doesn't play well with vitis ai quantization and compilation
        norm_layer=torch.nn.BatchNorm2d                 # vitis ai does not recognize frozen BN layer
    )
    head = Head(backbone.out_channels, 9, args.num_classes)
    model = Model(backbone, args.num_classes, head=head)
    model.eval().to(DEVICE)
    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location=DEVICE), strict=False)

    sample_inputs = torch.randn((1,3,args.img_h,args.img_w))
    mode = "calib" if args.command == "calibrate" else "test"

    quantizer = torch_quantizer(mode, model, sample_inputs, output_dir=args.output_dir)
    quant_model = quantizer.quant_model
    quant_model.eval()

    return quant_model, model, quantizer


def get_dataloader(args):
    transform = T.Compose([
        T.Resize((args.img_h,args.img_w)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    ds = ImageFolder(args.data_dir, transform=transform)
    if args.command == "export":
        args.num_samples = 1
    
    if args.num_samples > 0 and args.num_samples < len(ds):
        ds = Subset(ds, list(range(args.num_samples)))

    dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    return dataloader


@torch.no_grad()
def validate(model, dataloader):
    raise NotImplementedError()

    model.eval()
    
    for images, labels in tqdm(dataloader):
        images = images.to(DEVICE)
        outputs = model(images)
        

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str)
    
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--weights", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="./retinanet")
    parser.add_argument("--data_dir", type=str, default="./data")

    parser.add_argument("--img_w", type=int, default=640)
    parser.add_argument("--img_h", type=int, default=640)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=None)

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    assert args.command in ("calibrate", "test", "export")

    quant_model, float_model, quantizer = get_quantized_model(args)
    dataloader= get_dataloader(args)


    if args.command == "calibrate":
        for images, _ in tqdm(dataloader):
            images = images.to(DEVICE)
            with torch.no_grad():
                quant_model(images)

        quantizer.export_quant_config()

    elif args.command == "test":
        quant_map = validate(quant_model, dataloader)
        print(f"Quantized model mAP: {quant_map*100:.2f}")

        float_map = validate(float_model, dataloader)
        print(f"Original model mAP: {float_map*100:.2f}")

    else:
        img, _ = next(iter(dataloader))
        img = img.to(DEVICE)
        with torch.no_grad():
            quant_model(img)

        quantizer.export_xmodel(output_dir=args.output_dir)
