# CenterNet with KV260

`model.py` provides model definition for CenterNet. Only ResNet backbones are supported. The heads are fixed at 256-channels wide and 3-layer deep. FPN is also fixed.

## Model training

Model is trained from this repo: https://github.com/gau-nernst/centernet-lightning

## Inference

Only video file is supported for now

```bash
python inference.py macaque_test_01.mp4 --weights centernet_r34_fpn.pth --threshold 0.1 --img_w 960 --img_h 544
```

Dependencies: numpy, torch, torchvision, tqdm, opencv-python, albumentations, pycocotools, PIL.
