# Macaque detection

## Datasets

http://www.pri.kyoto-u.ac.jp/datasets/macaquepose/index.html

To extract bounding boxes from segmentation annotations and export them to YOLO format:

```bash
python ./scripts/macaquepose_v1_bboxes.py --data_dir PATH/TO/macaquepose_v1 --save_dir SAVE/DIRECTORY
```

`--data_dir` should contain `annotations.csv` file and `images` folder. `pandas` and `PIL` are required.
