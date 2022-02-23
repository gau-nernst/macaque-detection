# CenterNet with KV260

Go to [centernet_training](../centernet_training/) for more details about the model and how to train it.

Files overview:

- `model.py`: model definition for CenterNet. Depends on [vision_toolbox](https://github.com/gau-nernst/vision-toolbox) for backbones and FPN neck.
- `quantize.py`
- `data.py`

## Environment setup

Please use [Vitis AI 2.0](https://github.com/Xilinx/Vitis-AI) since it uses PyTorch 1.7.1 by default. GPU version is recommended.

Inside Vitis AI 2.0 Docker environment:

```bash
conda activate vitis-ai-pytorch
git clone https://github.com/gau-nernst/macaque-detection
cd macaque-detection/kv260_centernet
pip install -r requirements.txt --user
```

## Quantization

Calibration

```bash
python quantize.py --weights macaque_centernet_darknet.pth  --output_dir "centernet_darknet" --data_dir /datasets/NTU_macaque_videos/images/ --batch_size 64 calibrate
```

Test (validation)

```bash
python quantize.py --weights macaque_centernet_darknet.pth  --output_dir "centernet_darknet" --data_dir /datasets/NTU_macaque_videos/images/ --ann_json /datasets/NTU_macaque_videos/ntu_macaques_val.json --batch_size 64 test
```

Export

```bash
python quantize.py --weights macaque_centernet_darknet.pth  --output_dir "centernet_darknet" --data_dir /datasets/NTU_macaque_videos/images/ --ann_json /datasets/NTU_macaque_videos/ntu_macaques_val.json --batch_size 64 export
```

## Inference

Only video file is supported for now

```bash
python inference.py macaque_test_01.mp4 --weights centernet_r34_fpn.pth --threshold 0.1 --img_w 960 --img_h 544
```

Dependencies: numpy, torch, torchvision, tqdm, opencv-python, albumentations, pycocotools, Pillow
