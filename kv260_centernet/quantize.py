import argparse

from pytorch_nndct.apis import torch_quantizer

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data._utils.collate import default_collate
import torchvision.transforms as T
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import CenterNet, decode_detections
from data import ImageFolder, CocoDetection, collate_fn, CocoEvaluator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_quantized_model(args):
    model = CenterNet(1, "resnet34").eval()
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))

    sample_inputs = torch.randn((1,3,args.img_h,args.img_w))
    mode = "calib" if args.command == "calibrate" else "test"

    quantizer = torch_quantizer(mode, model, sample_inputs, output_dir=args.output_dir)
    quant_model = quantizer.quant_model
    quant_model.eval()

    return quant_model, model, quantizer


def get_dataloader(args):
    if args.command == "test":
        transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(),
            ToTensorV2()
        ], bbox_params=dict(format="coco", label_fields=["labels"], min_area=1))
        ds = CocoDetection(args.data_dir, args.ann_json, transforms=transform)
        collate = collate_fn

    else:
        transform = T.Compose([
            T.Resize((args.img_h,args.img_w)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        ds = ImageFolder(args.data_dir, transform=transform)
        collate = default_collate
    
    if args.command == "export":
        args.num_samples = 1
    
    if args.num_samples > 0 and args.num_samples < len(ds):
        ds = Subset(ds, list(range(args.num_samples)))

    dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate)
    return dataloader


@torch.no_grad()
def validate(model: CenterNet, dataloader, num_classes):
    model.eval()
    evaluator = CocoEvaluator(num_classes)
    
    for images, targets in tqdm(dataloader):
        images = images.to(DEVICE)
        heatmap, box_offsets = model(images)
        detections = decode_detections(heatmap.sigmoid(), box_offsets)

        detections = {k: v.cpu().numpy() for k, v in detections.items()}
        detections["boxes"][...,[0,2]] *= images.shape[2]               # convert to input images coordinates
        detections["boxes"][...,[1,3]] *= images.shape[1]
        detections["boxes"][...,2] -= detections["boxes"][...,0]
        detections["boxes"][...,3] -= detections["boxes"][...,1]
        keys = detections.keys()
        detections = [{k: detections[k][i] for k in keys} for i in range(len(detections["labels"]))]

        targets = [{k: np.array(v) for k, v in target.items()} for target in targets]
        evaluator.update(detections, targets)

    return evaluator.get_metrics()
    

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str)
    
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--weights", type=str)
    parser.add_argument("--output_dir", type=str, default="./centernet")
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--ann_json", type=str)

    parser.add_argument("--img_w", type=int, default=512)
    parser.add_argument("--img_h", type=int, default=512)

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
        quant_metrics = validate(quant_model, dataloader, args.num_classes)
        print(f"Quantized model mAP: {quant_metrics['mAP']*100:.2f}")

        float_metrics = validate(float_model, dataloader, args.num_classes)
        print(f"Original model mAP: {float_metrics['mAP']*100:.2f}")

    else:
        img, _ = next(iter(dataloader))
        img = img.to(DEVICE)
        with torch.no_grad():
            quant_model(img)

        quantizer.export_xmodel(output_dir=args.output_dir)
