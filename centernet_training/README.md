# CenterNet training

CenterNet with CSPDarknet YOLOv5m backbone and FPN neck

## Environment setup

Repo: [centernet-lightning](https://github.com/gau-nernst/centernet-lightning)

```bash
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
pip install pytorch-lightning pycocotools albumentations jsonargparse[signatures]
pip install git+https://github.com/gau-nernst/vision-toolbox.git
```

## Training

Use the config file `macaque.yaml` in this directory to train (copy it to `configs/` folder of the centernet-lightning repo).

```bash
git clone git+https://github.com/gau-nernst/centernet-lightning.git
cd centernet-lightning
python train.py fit --configs/centernet.yaml --configs configs/macaque.yaml
```

## Export weights for Vitis AI

Extract state dict from the saved checkpoint and rename the keys:

```python
import torch

state_dict = torch.load('checkpoint.ckpt', map_location='cpu')['state_dict']
state_dict = {k[len('model.'):]: v for k, v in state_dict.items()}

torch.save(state_dict, 'model_weights.pth')
```
