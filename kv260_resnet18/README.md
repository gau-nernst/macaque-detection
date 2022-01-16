# ResNet18 on KV260

Example on how to run PyTorch ResNet18 on KV260.

## Environment

### Flash board image to KV260

Follow the instruction here: https://github.com/Xilinx/Vitis-AI/tree/master/setup/mpsoc/VART

- Download board image: https://www.xilinx.com/member/forms/download/design-license-xef.html?filename=xilinx-kv260-dpu-v2020.2-v1.4.0.img.gz
- Use [Etcher](https://www.balena.io/etcher/) to flash the image to your micro SD card (recommend to use the portable version)
- Note: this board image is configured for B4096 architecture, and uses Vitis-AI 1.4.0

### Training environment (to train the model)

- Only PyTorch and torchvision required
- [Official instruction](https://pytorch.org/get-started/locally/)
- Version used in this example: PyTorch 1.10.0, torchvision 0.11.1
- Example (with conda)

```bash
conda create -n pt python=3.8
conda activate pt
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
```

Vitis AI environment (to compile the model)

- Install via Docker
- [Official instruction](https://github.com/Xilinx/Vitis-AI)
- GPU Docker image is not required
- Docker with WSL also works
- Note: use Vitis-AI 1.4.0 so that it matches the board image. 1.4.1 might work but we haven't tested.
- Example (assumed Docker is installed)

```bash
wget https://github.com/Xilinx/Vitis-AI/archive/refs/tags/v1.4.zip  # download Vitis-AI
unzip Vitis-AI-1.4.zip
docker pull xilinx/vitis-ai:1.4.916                 # download Vitis-AI Docker image
```

## Dataset

Any image classification dataset is fine. In this example, we will use a small 'Cat and Dog' dataset from Kaggle. Link: https://www.kaggle.com/tongpython/cat-and-dog

```bash
kaggle datasets download -d tongpython/cat-and-dog
unzip cat-and-dog.zip -d cat-and-dog
```

Assume the data is in `data/training_set/training_set` and `data/test_set/test_set` for train set and validation set respectively.

## Train

`train.py` is a standard Transfer learning for Image classification script. For more information about Transfer learning in PyTorch, see [here](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).

In your training environment, run `train.py`:

```bash
python train.py --num_classes 2 --train_dir "data/training_set/training_set" --val_dir "data/test_set/test_set" --lr 0.001 --num_epochs 10 --batch_size 64
```

This will automatically save the best weights in the current folder e.g. `epoch-006-acc-99.26.pth`.

Note:

- The train script was only tested with PyTorch 1.10.0 and torchvision 0.11.1
- Since Vitis AI Docker uses an old version of PyTorch (1.4.0), you have to pass `_use_new_zipfile_serialization=False` in `torch.save()` so that we can load the weights in Vitis-AI Docker later.

## Create xmodel

You will need a dataset for quantization calibration, and optionally for testing, inside the Docker container. You can either copy your dataset to the `Vitis-AI-1.4/` folder, or you can mount the dataset folder in the container.

To mount your dataset folder in the container, open the file `docker_run.sh` in `Vitis-AI-1.4/` and add mount options in the Docker command:

```bash
docker_run_params=$(cat <<-END)
    -v /absolute/path/to/dataset:/dataset \         # add this line
    ...                                             # other stuff that was here
END
)
```

Assume that `/absolute/path/to/dataset` points the test set of the Cat and Dog dataset we downloaded earlier (`cat-and-dog/test_set/test_set`). You should also copy the file `quantize.py` in this folder, and the trained weights `epoch-006-acc-99.26.pth`, to the `Vitis-AI-1.4/` folder.

Enter Docker and activate the PyTorch environment

```bash
cd Vitis-AI-1.4
./docker_run.sh                     # click through the EULA and press y
conda activate vitis-ai-pytorch     # activate pytorch environment
```

Now your dataset should be present under `/dataset`.

There are 3 steps

1. Quantize the model. A calibration dataset is needed for this. Labels are not needed.
2. (Optional) Evaluate the quantized model and compare it with the original (float) model.
3. Compile the model

The first 2 steps are covered in `resnet18.py`.

### Quantize

```bash
python resnet18.py calibrate --weights model.pth --data_dir /dataset --num_samples 100 --output_dir resnet18
```

This will take some minutes. When this finishes, you will have `ResNet.py`, `quant_info.json`, and `bias_corr.pth`.

Now you can export the quantized model to xmodel format (`output_dir` must be the same as above). You will have `ResNet_int.xmodel`.

```bash
python resnet18.py export --weights model.pth --data_dir /dataset --output_dir resnet18
```

### (Optional) Evaluate

Again, `output_dir` must be the same as above so that Xilinx can obtain information about the quantized model.

```bash
python resnet18.py test --weights model.pth --data_dir /dataset --num_samples 500 --output_dir resnet18
```

### Compile

```bash
vai_c_xir --xmodel "resnet18/ResNet_int.xmodel" --arch "/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json" --net_name r18_kv260 --output_dir "resnet18/compile"
```

This will produce `r18_kv260.xmodel`, `meta.json`, and `md5sum.txt`.

Sample output

```bash
**************************************************
* VITIS_AI Compilation - Xilinx Inc.
**************************************************
[UNILOG][INFO] Target architecture: DPUCZDX8G_ISA0_B4096_MAX_BG2
[UNILOG][INFO] Compile mode: dpu
[UNILOG][INFO] Debug mode: function
[UNILOG][INFO] Target architecture: DPUCZDX8G_ISA0_B4096_MAX_BG2
[UNILOG][INFO] Graph name: ResNet, with op num: 169
[UNILOG][INFO] Begin to compile...
[UNILOG][INFO] Total device subgraph number 3, DPU subgraph number 1
[UNILOG][INFO] Compile done.
[UNILOG][INFO] The meta json is saved to "/macaque-detection/kv260_resnet18/resnet18/compile/meta.json"
[UNILOG][INFO] The compiled xmodel is saved to "/macaque-detection/kv260_resnet18/resnet18/compile/r18_kv260.xmodel"
[UNILOG][INFO] The compiled xmodel's md5sum is d8aa2a6f25878e1b6f4f982f2cc92a79, and has been saved to "/macaque-detection/kv260_resnet18/resnet18/compile/md5sum.txt"
```

## Run ResNet18 on KV260

Assume KV260's IP address is `192.168.1.10`. We only need to copy the compiled xmodel `r18_kv260.xmodel` and sample Python script `app.py` to the board (`app.py` is in this folder). Run the commands below in the host machine, not inside the Docker container.

```bash
scp app.py root@192.168.1.10:~/
scp Vitis-AI-1.4/resnet18/compile/r18_kv260.xmodel root@192.168.1.10:~/
```

You can also use VS Code to open an SSH window connecting to the KV260 board. Then you can drag and drop the files to KV260.

You will also need to copy a sample image to KV260. Assume we have a 224 x 224 image of a cat `cat_224.jpg` copied to the board. Connect to KV260 via ssh and run the app

```bash
ssh root@192.168.1.10

# inside KV260 now
python3 app.py --xmodel r18_kv260.xmodel --image cat_224.jpg
```
