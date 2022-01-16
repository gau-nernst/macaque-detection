import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision.models import resnet
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
try:
    from tqdm import tqdm
except:
    tqdm = None


def get_train_dataloader(args):
    transform = T.Compose([
        T.RandomResizedCrop(224),
        T.ColorJitter(0.2, 0.2, 0.2),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    ds = ImageFolder(args.train_dir, transform=transform)
    dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    return dataloader


def get_val_dataloader(args):
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    ds = ImageFolder(args.val_dir, transform=transform)
    dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return dataloader


def train_one_epoch(model: torch.nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.to(device)
    model.train()

    iterator = tqdm(dataloader) if tqdm is not None else dataloader
    for (images, labels) in iterator:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


@torch.inference_mode()
def validate(model: torch.nn.Module, dataloader, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    N = 0
    correct = 0
    for (images, labels) in dataloader:
        images = images.to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=-1)

        N += len(labels)
        correct += (preds.cpu() == labels).sum().item()

    acc = correct / N
    return acc


def get_args_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--train_dir", type=str)
    parser.add_argument("--val_dir", type=str)

    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    
    return parser


def main(args):
    m = resnet.__dict__[args.model](pretrained=True)
    m.fc = torch.nn.LazyLinear(args.num_classes)

    train_dataloader = get_train_dataloader(args)
    val_dataloader = get_val_dataloader(args)

    optimizer = torch.optim.SGD(m.parameters(), args.lr, momentum=0.9)
    
    main_scheduler = CosineAnnealingLR(optimizer, args.num_epochs - 5)
    warmup_scheduler = LinearLR(optimizer, 0.01, total_iters=5)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, main_scheduler], milestones=[5])

    best_acc = 0
    template_name = "{model}_epoch-{epoch:03d}_acc-{acc:.2f}.pth"
    best_name = template_name.format(model=args.model, epoch=-1, acc=0)
    for i in range(args.num_epochs):
        print(f"Epoch {i}")
        train_one_epoch(m, train_dataloader, optimizer)
        scheduler.step()
    
        acc = validate(m, val_dataloader)
        print(f"Acc: {acc*100:.2f}")

        if acc > best_acc:
            best_acc = acc
            if os.path.exists(best_name):
                os.remove(best_name)

            best_name = template_name.format(model=args.model, epoch=i, acc=acc*100)
            torch.save(m.state_dict(), best_name)

    print(f"Best acc: {best_acc*100:.2f}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
