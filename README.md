# Macaque detection

## Datasets

http://www.pri.kyoto-u.ac.jp/datasets/macaquepose/index.html

Prefix | Description | Count
-------|-------------|-------
PRI    | Primate Research Institute, Kyoto University (Species: Japanese Macaque) | 1641
ZooA   | Toyama Municipal Family Park Zoo (Species: Japanese Macaque) | 3784
ZooB   | Itozu no Mori Zoological Park (Species: Japanese Macaque) | 1312
ZooC   | Inokashira Park Zoo (Species: Rhesus Macaque) | 2747
ZooD   | Tobu Zoo (Species: Rhesus Macaque) | 2461
Other  | Google Open Images Dataset (Species: Various) | 1138
Total  | Total | 13083

To extract bounding boxes from segmentation annotations and export them to COCO format (YOLO is also supported):

```bash
python scripts/macaquepose_v1_bboxes.py --data_dir PATH/TO/macaquepose_v1 --output train.json --split train --format coco
python scripts/macaquepose_v1_bboxes.py --data_dir PATH/TO/macaquepose_v1 --output val.json --split val --format coco
python scripts/macaquepose_v1_bboxes.py --data_dir PATH/TO/macaquepose_v1 --output all.json --split all --format coco
```

Notes:

- `--data_dir` should contain `annotations.csv` file and `images` folder. `pandas` and `PIL` are required.
- If YOLO format is used, `--output` should be a subdirectory name.
- Train split: PRI, ZooA, ZooB, ZooC, ZooD. Val split: Other (Open Images).

## Training

Training script is adapted from [torchvision](https://github.com/pytorch/vision/tree/main/references/detection). Some default values are changed. We use RetinaNet with ResNet-50 FPN backbone.

```bash
python detection/train.py --data-path datasets/macaquepose_v1 -b 16 --epochs 5 --pretrained --amp 
```
