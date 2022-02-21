# CenterNet training

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

TBD
