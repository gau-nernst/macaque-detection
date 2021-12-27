import argparse

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torchvision.models import resnet18
from pytorch_nndct.apis import torch_quantizer
from tqdm import tqdm


def get_quantized_model(args):
    model = resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, args.num_classes)
    model.load_state_dict(torch.load(args.weights_path, map_location="cpu"))
    model.eval()

    sample_inputs = torch.randn((1,3,224,224))
    mode = "calib" if args.command == "calibrate" else "test"

    quantizer = torch_quantizer(mode, model, sample_inputs, output_dir=args.output_dir)
    quant_model = quantizer.quant_model
    quant_model.eval()

    return quant_model, model, quantizer


def get_dataloader(args):
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    ds = ImageFolder(args.data_dir, transform=transform)
    if args.command == "export":
        args.num_samples = 1
    
    if args.num_samples > 0 and args.num_samples < len(ds):
        ds = Subset(ds, list(range(args.num_samples)))

    dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    return dataloader


@torch.no_grad()
def validate(model, dataloader):
    model.eval()
    N = 0
    correct = 0

    for images, labels in tqdm(dataloader):
        outputs = model(images)
        preds = torch.argmax(outputs, dim=-1)

        N += len(labels)
        correct += (preds == labels).sum().item()

    acc = correct / N
    return acc


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str)
    
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--weights_path", type=str, default="./model.pth")
    parser.add_argument("--output_dir", type=str, default="./resnet18")
    parser.add_argument("--data_dir", type=str, default="./data")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=None)

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    assert args.command in ("calibrate", "test", "export")

    quant_model, float_model, quantizer = get_quantized_model(args)
    dataloader= get_dataloader(args)


    if args.command == "calibrate":
        for images, _ in tqdm(dataloader):
            with torch.no_grad():
                quant_model(images)

        quantizer.export_quant_config()

    elif args.command == "test":
        quant_acc = validate(quant_model, dataloader)
        print(f"Quantized model acc: {quant_acc*100:.2f}")

        float_acc = validate(float_model, dataloader)
        print(f"Original model acc: {float_acc*100:.2f}")

    else:
        img, _ = next(iter(dataloader))
        with torch.no_grad():
            quant_model(img)

        quantizer.export_xmodel(output_dir=args.output_dir)
