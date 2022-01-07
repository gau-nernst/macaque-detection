import argparse
import os
import random

import torch
from torch.utils.data import Subset, DataLoader
from torchvision import models
import torchvision.transforms as T
from torchvision.transforms.functional import convert_image_dtype
import cv2
from tqdm import tqdm

import utils
from coco_utils import get_coco
from presets import DetectionPresetEval


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


@torch.inference_mode()
def visualize(model, dataloader, device="cuda", save_dir=None):
    model.eval().to(device)

    for images, targets in tqdm(dataloader):
        images_cpu = [convert_image_dtype(img, torch.uint8).numpy().transpose(1,2,0) for img in images]
        targets = [{k: v.numpy() for k, v in t.items()} for t in targets]
        
        images = list(img.to(device) for img in images)
        outputs = model(images)
        outputs = [{k: v.cpu().numpy() for k, v in t.items()} for t in outputs]
        
        # targets keys: boxes, labels, image_id, area, iscrowd
        # outputs keys: boxes, scores, labels
        for img, target, output in zip(images_cpu, targets, outputs):
            target_texts = ["ground truth"] * len(target["boxes"])
            output_texts = [f"prediction: {round(x*100)}%" for x in output["scores"]]
            
            draw_bboxes(img, target["boxes"], target_texts, text_top=True, color=GREEN, text_color=BLACK, inplace=True)
            draw_bboxes(img, output["boxes"], output_texts, text_top=False, color=RED, text_color=WHITE, inplace=True)

            if save_dir is not None:
                cv2.imwrite(os.path.join(save_dir, f"{target['image_id']}.jpg"), img[...,::-1])


def main(args):
    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)

    dataset = get_coco(args.data_path, "val", "images", DetectionPresetEval())
    ds_size = len(dataset)
    if args.num_samples < ds_size:
        indices = random.sample(range(ds_size), args.num_samples) if args.shuffle else list(range(args.num_samples))
        dataset = Subset(dataset, indices)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)
    
    model = models.detection.__dict__[args.model](num_classes=1, pretrained=False, pretrained_backbone=False, score_thresh=0.3)
    state_dict = torch.load(args.checkpoint, map_location="cpu")["model"]
    model.load_state_dict(state_dict)
    
    visualize(model, dataloader, device=args.device, save_dir=args.save_dir)


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)

    parser.add_argument("--model", type=str, default="retinanet_resnet50_fpn")
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pth")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--save_dir", type=str, default=None)

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
