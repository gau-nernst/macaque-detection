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

def get_box_stats(bboxes):
    ratios = []
    sizes = []
    for x1, y1, x2, y2 in bboxes:
        w = x2 - x1
        h = y2 - y1
        ratios.append(w/h)
        sizes.append((w*h)**0.5)

def get_image_stats(img_paths):
    ratios = []
    sizes = []
    for path in img_paths:
        img_w, img_h = Image.open(path).size
        ratios.append(img_w / img_h)
        sizes.append((img_w*img_h)**0.5)

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
    # https://github.com/pytorch/vision/issues/1530
    ann_id = 1
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
                "id": ann_id,
                "image_id": i,
                "category_id": 0,
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "iscrowd": 0
            })
            ann_id += 1
    
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
    parser.add_argument("--split", type=str, default="train")
    
    args = parser.parse_args()
    assert args.format in ("yolo", "coco")

    csv_path = os.path.join(args.data_dir, "annotations.csv")
    img_dir = os.path.join(args.data_dir, "images")

    df = pd.read_csv(csv_path)

    # train-val split
    # train: PRI, ZooA, ZooB, ZooC, ZooD
    # val: OpenImages
    df_train = df[df["image file name"].str.startswith(("PRI", "Zoo"))]
    df_val = df[~df.index.isin(df_train.index)]
    
    choices = {
        "all": df,
        "train": df_train,
        "val": df_val
    }
    df_out = choices[args.split]
    bboxes = df_out["segmentation"].apply(json.loads).apply(instance_segmentations_to_bboxes)

    output = os.path.join(args.data_dir, args.output)
    if args.format == "yolo":
        bboxes_to_yolo(df_out["image file name"].tolist(), bboxes.tolist(), img_dir, output)
    else:
        bboxes_to_coco(df_out["image file name"].tolist(), bboxes.tolist(), img_dir, output)

if __name__ == "__main__":
    main()
