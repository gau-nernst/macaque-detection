# standard libraries
import argparse

# 3rd party libraries
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.utils.data._utils.collate import default_collate
import torchvision.transforms as T
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# local files
from model import build_model, decode_detections
from data import CalibrationDataset, CocoDetection, coco_collate, CocoEvaluator

# Vitis AI
from pytorch_nndct.apis import torch_quantizer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_quantized_model(args):
    model = build_model(args.weights, 'darknet_yolov5m')

    sample_inputs = torch.randn((1,3,args.img_h,args.img_w))
    mode = "calib" if args.command == "calibrate" else "test"

    quantizer = torch_quantizer(mode, model, sample_inputs, output_dir=args.output_dir)
    quant_model = quantizer.quant_model
    quant_model.eval()

    return quant_model, model, quantizer


def get_dataset(data_dir, img_h=512, img_w=512, ann_json=None, detection=False):
    if detection:
        transform = A.Compose([
            A.SmallestMaxSize(max(args.img_h, args.img_w)),
            A.CenterCrop(args.img_h, args.img_w),
            A.Normalize(mean=(0,0,0), std=(1,1,1)),
            ToTensorV2()
        ], bbox_params=dict(format="coco", label_fields=["labels"], min_area=1))
        ds = CocoDetection(data_dir, ann_json, transforms=transform)

    else:
        transform = T.Compose([
            T.Resize(max(args.img_h, args.img_w)),
            T.CenterCrop((args.img_h, args.img_w)),
            T.ToTensor(),
            T.Normalize(mean=(0,0,0), std=(1,1,1))
        ])
        ds = CalibrationDataset(data_dir, transforms=transform)
    
    return ds


@torch.no_grad()
def validate(model, dataloader, num_classes):
    model.eval()
    evaluator = CocoEvaluator(num_classes)
    
    for images, targets in tqdm(dataloader):
        images = images.to(DEVICE)
        heatmap, box_offsets = model(images)
        heatmap = heatmap.sigmoid()
        detections = decode_detections(heatmap, box_offsets)

        detections = {k: v.cpu().numpy() for k, v in detections.items()}
        detections["boxes"][...,[0,2]] *= images.shape[3]               # convert to input images coordinates
        detections["boxes"][...,[1,3]] *= images.shape[2]
        detections["boxes"][...,2] -= detections["boxes"][...,0]
        detections["boxes"][...,3] -= detections["boxes"][...,1]
        keys = detections.keys()
        detections = [{k: detections[k][i] for k in keys} for i in range(len(detections["labels"]))]

        targets = [{k: np.array(v) for k, v in target.items()} for target in targets]
        evaluator.update(detections, targets)

    return evaluator.get_metrics()


def main(args):
    assert args.command in ("calibrate", "test", "export")
    if args.command == 'export':
        args.num_samples = 1
    detection = args.command == 'test'
    collate_fn = coco_collate if detection else default_collate

    quant_model, float_model, quantizer = get_quantized_model(args)
    dataset = get_dataset(
        args.data_dir, args.img_h, args.img_w, args.ann_json, detection=detection
    )
    if 0 < args.num_samples < len(dataset):
        dataset = Subset(dataset, list(range(args.num_samples)))
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=4,
        collate_fn=collate_fn, pin_memory=True
    )

    if args.command == "calibrate":
        for images in tqdm(dataloader):
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
        img = next(iter(dataloader))
        img = img.to(DEVICE)
        with torch.no_grad():
            quant_model(img)

        quantizer.export_xmodel(output_dir=args.output_dir)


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str)
    
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--weights")
    parser.add_argument("--output_dir", default="./centernet")
    parser.add_argument("--data_dir")
    parser.add_argument("--ann_json")

    parser.add_argument("--img_w", type=int, default=512)
    parser.add_argument("--img_h", type=int, default=512)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=0)

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
