import argparse
import os

import numpy as np
import torch
import cv2
from tqdm import tqdm

from model import CenterNet
from quantize import get_quantized_model, get_dataloader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
WHITE = (255,255,255)
BLACK = (0,0,0)

def draw_bboxes(image, boxes, texts=None, text_top=True, color=RED, text_color=WHITE, inplace=False):
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
def visualize(model: CenterNet, dataloader, save_dir):
    model.eval()

    for images, targets in tqdm(dataloader):
        images_np = images.clone().numpy()
        heatmap, box_offsets = model(images.to(DEVICE))
        detections = model.decode_detections(heatmap, box_offsets)
        detections = {k: v.cpu().numpy() for k, v in detections.items()}

        for img, boxes, scores, labels, target in zip(images_np, detections["boxes"], detections["scores"], detections["labels"], targets):
            img_w = target["image_width"]
            img_h = target["image_height"]
            gt_boxes = np.array(target["boxes"])
            gt_labels = np.array(target["labels"])

            boxes[...,[0,2]] *= img_w / img.shape[-1]
            boxes[...,[1,3]] *= img_h / img.shape[-2]


            img = cv2.resize(img, (img_w, img_h))


def get_args_parser():
    parser = argparse.ArgumentParser()    
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--weights", type=str)
    parser.add_argument("--output_dir", type=str, default="centernet")
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--ann_json", type=str)

    parser.add_argument("--img_w", type=int, default=512)
    parser.add_argument("--img_h", type=int, default=512)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--quant_model", action="store_true")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()

    setattr(args, "command", "test")
    quant_model, float_model = get_quantized_model(args)
    dataloader = get_dataloader(args)

    # model = quant_model if args.quant_model else float_model
    visualize(float_model, dataloader, os.path.join(args.output_dir, "visualize"))
