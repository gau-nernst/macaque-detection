import argparse
import json
import os

import pandas as pd
from PIL import Image

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
    for name, img_bboxes in zip(img_names, bboxes):
        image_path = os.path.join(img_dir, name)
        label_path = os.path.join(save_dir, f"{os.path.splitext(name)[0]}.txt")

        img = Image.open(image_path)
        img_w, img_h = img.size

        with open(label_path, "w") as f:
            for x1, y1, x2, y2 in img_bboxes:
                cx = (x1 + x2) / 2 / img_w
                cy = (y1 + y2) / 2 / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h

                f.write(f"0 {cx} {cy} {w} {h}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--save_dir", type=str)
    args = parser.parse_args()

    csv_path = os.path.join(args.data_dir, "annotations.csv")
    img_dir = os.path.join(args.data_dir, "images")

    df = pd.read_csv(csv_path)
    bboxes = df["segmentation"].apply(json.loads).apply(instance_segmentations_to_bboxes)
    bboxes_to_yolo(df["image file name"].tolist(), bboxes.tolist(), img_dir, args.save_dir)

if __name__ == "__main__":
    main()
