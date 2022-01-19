import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


class ConvBnAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)


class ResNetBackbone(nn.Module):
    def __init__(self, name):
        super().__init__()
        m = models.__dict__[name]().eval()
        self.feat_extractor = nn.Module()
        for name, module in m.named_children():
            if "fc" not in name:
                self.feat_extractor.add_module(name, module)

        with torch.no_grad():
            self.out_channels = [x.shape[1] for x in self(torch.rand(1,3,224,224))]

    def forward(self, x):
        out1 = self.feat_extractor.conv1(x)
        out1 = self.feat_extractor.bn1(out1)
        out1 = self.feat_extractor.maxpool(out1)
        out2 = self.feat_extractor.layer1(out1)
        out3 = self.feat_extractor.layer2(out2)
        out4 = self.feat_extractor.layer3(out3)
        out5 = self.feat_extractor.layer4(out4)
        return [out2, out3, out4, out5]


class FPN(nn.Module):
    def __init__(self, in_channels, out_channels=256, block=ConvBnAct):
        super().__init__()
        self.out_channels = out_channels
        self.stride = 2**(len(in_channels)-1)

        self.lateral_convs = nn.ModuleList([nn.Conv2d(in_c, out_channels, kernel_size=1) for in_c in in_channels])
        self.output_convs = nn.ModuleList([block(out_channels, out_channels) for _ in range(len(in_channels)-1)])

    def forward(self, x):
        laterals = [l_conv(x[i]) for i, l_conv in enumerate(self.lateral_convs)]
        outputs = [laterals.pop()]
        
        for o_conv in self.output_convs:
            out = F.interpolate(outputs[-1], scale_factor=2., mode="nearest")
            out = out + laterals.pop()
            outputs.append(o_conv(out))

        return outputs[-1]


class Head(nn.Module):
    def __init__(self, in_channels, out_channels, width=256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, out_channels, 1)
        )
    
    def forward(self, x):
        return self.layers(x)


class CenterNet(nn.Module):
    def __init__(self, num_classes, backbone):
        super().__init__()
        self.backbone = ResNetBackbone(backbone)
        self.neck = FPN(self.backbone.out_channels)
        self.heads = nn.ModuleDict({
            "heatmap": Head(self.neck.out_channels, num_classes),
            "box_2d": Head(self.neck.out_channels, 4)
        })

    def forward(self, x):
        out = self.backbone(x)
        out = self.neck(out)
        # return self.heads.heatmap(out), self.heads.box_2d(out)
        return torch.cat((self.heads.heatmap(out), self.heads.box_2d(out)), dim=1)
