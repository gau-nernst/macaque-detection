import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import cv2
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import CenterNet, decode_detections
from data import CocoDetection, collate_fn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_model(args):
    model = CenterNet(1, "resnet34").eval()
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))

    sample_inputs = torch.randn((1,3,args.img_h,args.img_w))
    if args.quant_model:
        # lazy import, so that this script can run outside Vitis AI for float model
        from pytorch_nndct.apis import torch_quantizer
        
        quantizer = torch_quantizer("test", model, sample_inputs, output_dir=args.output_dir)
        model = quantizer.quant_model
        model.eval()

    model.to(DEVICE)
    return model


imagenet_mean = np.array((0.485, 0.456, 0.406))
imagenet_std = np.array((0.229, 0.224, 0.225))

RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
WHITE = (255,255,255)
BLACK = (0,0,0)

def draw_boxes(image, boxes, texts=None, text_top=True, color=RED, text_color=WHITE, inplace=False):
    box_thickness = 5
    font = cv2.FONT_HERSHEY_DUPLEX
    text_size = 2
    text_thickness = 2

    if not inplace:
        image = image.copy()
    
    boxes = [[round(x) for x in box] for box in boxes]
    if texts is None:
        for x1,y1,x2,y2 in boxes:
            cv2.rectangle(image, (x1,y1), (x2,y2), color, thickness=box_thickness)
    
    else:
        for (x1,y1,x2,y2), text in zip(boxes, texts):
            cv2.rectangle(image, (x1,y1), (x2, y2), color, thickness=box_thickness)
            
            (text_w, text_h), baseline = cv2.getTextSize(text, font, text_size, text_thickness)
            if text_top:
                text_pt = (x1, y1 + text_h)
                pt1 = (x1, y1 + text_h + baseline)
                pt2 = (x1 + text_w, y1)
            else:
                text_pt = (x1, y2 - baseline)
                pt1 = (x1, y2)
                pt2 = (x1 + text_w, y2 - text_h - baseline)
            cv2.rectangle(image, pt1, pt2, color, thickness=cv2.FILLED)
            cv2.putText(image, text, text_pt, font, text_size, text_color, thickness=text_thickness)

    return image


@torch.no_grad()
def visualize(model: CenterNet, dataloader, save_dir, threshold=0.3):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    count = 1
    for images, targets in tqdm(dataloader):
        images_np = images.clone().numpy().transpose(0,2,3,1)
        images_np = images_np * imagenet_std.reshape(1,1,1,3) + imagenet_mean.reshape(1,1,1,3)
        images_np = (images_np * 255).astype(np.uint8)

        heatmap, box_offsets = model(images.to(DEVICE))
        detections = decode_detections(heatmap.sigmoid(), box_offsets)
        detections = {k: v.cpu().numpy() for k, v in detections.items()}

        for img, boxes, scores, target in zip(images_np, detections["boxes"], detections["scores"], targets):
            img_w = target["image_width"]                   # original sizes
            img_h = target["image_height"]
            gt_boxes = np.array(target["boxes"])
            gt_boxes[...,[0,2]] *= img_w / img.shape[1]     # convert to original sizes
            gt_boxes[...,[1,3]] *= img_h / img.shape[0]
            gt_boxes[...,[2,3]] += gt_boxes[...,[0,1]]      # xywh to xyxy
            gt_boxes = gt_boxes.round().astype(int)

            mask = scores > threshold                       # filter
            boxes = boxes[mask]
            scores = scores[mask]
            boxes[...,[0,2]] *= img_w                       # convert to original sizes
            boxes[...,[1,3]] *= img_h
            boxes = boxes.round().astype(int)

            img = cv2.resize(img, (img_w, img_h))
            draw_boxes(img, gt_boxes, texts=["ground truth"]*len(gt_boxes), text_top=True, color=GREEN, text_color=BLACK, inplace=True)
            draw_boxes(img, boxes, texts=[f"prediction: {x*100:.0f}%" for x in scores], text_top=False, color=RED, text_color=WHITE, inplace=True)

            cv2.imwrite(os.path.join(save_dir, f"image_{count:04d}.jpg"), img[...,::-1])
            count += 1


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--weights")
    parser.add_argument("--output_dir", default="centernet")
    parser.add_argument("--data_dir")
    parser.add_argument("--ann_json")

    parser.add_argument("--img_w", type=int, default=512)
    parser.add_argument("--img_h", type=int, default=512)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--quant_model", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.3)

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()

    # build dataloader
    transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(),
        ToTensorV2()
    ], bbox_params=dict(format="coco", label_fields=["labels"], min_area=1))
    ds = CocoDetection(args.data_dir, args.ann_json, transforms=transform)
    if args.num_samples < len(ds):
        ds = Subset(ds, list(range(args.num_samples)))
    dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)


    save_dir = 'quant' if args.quant_model else 'float'
    save_dir = os.path.join(args.output_dir, f"visualize_{save_dir}")

    model = get_model(args)
    visualize(model, dataloader, save_dir, threshold=args.threshold)
