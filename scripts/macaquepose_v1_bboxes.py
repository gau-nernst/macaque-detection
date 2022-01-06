import argparse
import json
import os

import pandas as pd
from PIL import Image
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

def instance_segmentations_to_bboxes(instances):
    bboxes = []
    for instance in instances:
        x1, x2, y1, y2 = float("inf"), 0, float("inf"), 0
        
        # an instance may have multiple disconnected segments
        for segment in instance:
            for x, y in segment["segment"]:
                x1 = min(x1, x)
                x2 = max(x2, x)
                y1 = min(y1, y)
                y2 = max(y2, y)
        
        bboxes.append((x1,y1,x2,y2))

    return bboxes

def bboxes_to_yolo(img_names, bboxes, img_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    iterator = zip(img_names, bboxes)
    if tqdm is not None:
        iterator = tqdm(iterator, total=len(img_names))

    for name, img_bboxes in iterator:
        img_path = os.path.join(img_dir, name)
        img_w, img_h = Image.open(img_path).size

        label_path = os.path.join(save_dir, f"{os.path.splitext(name)[0]}.txt")
        with open(label_path, "w") as f:
            for x1, y1, x2, y2 in img_bboxes:
                cx = (x1 + x2) / 2 / img_w
                cy = (y1 + y2) / 2 / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h

                f.write(f"0 {cx} {cy} {w} {h}\n")

def bboxes_to_coco(img_names, bboxes, img_dir, save_path):
    count = 0
    images = []
    annotations = []
    categories = [{"id": 0, "name": "macaque"}]

    iterator = zip(img_names, bboxes)
    if tqdm is not None:
        iterator = tqdm(iterator, total=len(img_names))

    for i, (name, img_bboxes) in enumerate(iterator):
        img_path = os.path.join(img_dir, name)
        img_w, img_h = Image.open(img_path).size
        images.append({
            "id": i,
            "width": img_w,
            "height": img_h,
            "file_name": name
        })

        for x1, y1, x2, y2 in img_bboxes:
            w = x2 - x1
            h = y2 - y1
            annotations.append({
                "id": count,
                "image_id": i,
                "category_id": 0,
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "iscrowd": 0
            })
    
    coco_ann = {
        "info": None,
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "licenses": None,
    }
    with open(save_path, "w") as f:
        json.dump(coco_ann, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--format", type=str, default="coco")
    
    args = parser.parse_args()
    assert args.format in ("yolo", "coco")

    csv_path = os.path.join(args.data_dir, "annotations.csv")
    img_dir = os.path.join(args.data_dir, "images")

    df = pd.read_csv(csv_path)
    bboxes = df["segmentation"].apply(json.loads).apply(instance_segmentations_to_bboxes)

    if args.format == "yolo":
        bboxes_to_yolo(df["image file name"].tolist(), bboxes.tolist(), img_dir, args.output)
    else:
        bboxes_to_coco(df["image file name"].tolist(), bboxes.tolist(), img_dir, args.output)

if __name__ == "__main__":
    main()
