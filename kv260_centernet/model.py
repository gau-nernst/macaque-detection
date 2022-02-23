import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from vision_toolbox import backbones, ConvBnAct, FPN


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
        return self.forward_features(x)[-1]

    def forward_features(self, x):
        out1 = self.feat_extractor.conv1(x)
        out1 = self.feat_extractor.bn1(out1)
        out1 = self.feat_extractor.maxpool(out1)
        out2 = self.feat_extractor.layer1(out1)
        out3 = self.feat_extractor.layer2(out2)
        out4 = self.feat_extractor.layer3(out3)
        out5 = self.feat_extractor.layer4(out4)
        return [out2, out3, out4, out5]


class Head(nn.Sequential):
    def __init__(self, in_channels, out_channels, width, depth):
        super().__init__()
        for i in range(depth):
            in_c = in_channels if i == 0 else width
            self.add_module(f'block_{i+1}', ConvBnAct(in_c, width))
        self.out_conv = nn.Conv2d(width, out_channels, 1)


class CenterNet(nn.Module):
    def __init__(self, backbone, neck, heads):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.heads = heads
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.backbone.forward_features(x)
        out = self.neck(out)
        heatmap = self.heads.heatmap(out)
        box_offsets = self.heads.box_2d(out)
        box_offsets = self.relu(box_offsets)
        return heatmap, box_offsets


def build_model(pth_path, backbone_name):
    state_dict = torch.load(pth_path, map_location='cpu')

    # backbone
    backbone = backbones.__dict__[backbone_name]()
    # backbone = ResNetBackbone(backbone_name)

    # neck
    neck_out_channels = state_dict['neck.lateral_convs.0.weight'].shape[0]
    neck = FPN(backbone.out_channels, out_channels=neck_out_channels)

    # heads
    num_classes = state_dict['heads.heatmap.out_conv.weight'].shape[0]
    head_width = state_dict['heads.heatmap.block_1.conv.weight'].shape[0]
    head_depth = len([x for x in state_dict.keys() if x.startswith('heads.heatmap.block_') and x.endswith('.conv.weight')])
    heads = nn.ModuleDict({
        "heatmap": Head(neck_out_channels, num_classes, head_width, head_depth),
        "box_2d": Head(neck_out_channels, 4, head_width, head_depth)
    })
    state_dict['heads.box_2d.out_conv.weight'] *= 16
    state_dict['heads.box_2d.out_conv.bias'] *= 16

    model = CenterNet(backbone, neck, heads).eval()
    model.load_state_dict(state_dict)
    model(torch.rand(1,3,512,512))
    return model


def decode_detections(heatmap, box_offsets):
    batch_size, _, out_h, out_w = heatmap.shape

    # 1. pseudo-nms via max pool
    nms_mask = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1) == heatmap
    heatmap = heatmap * nms_mask
    
    # 2. since box regression is shared, we only consider the best candidate at each heatmap location
    heatmap, labels = torch.max(heatmap, dim=1)

    # 3. flatten and get topk
    heatmap = heatmap.view(batch_size, -1)
    labels = labels.view(batch_size, -1)
    scores, indices = torch.topk(heatmap, 100)
    labels = torch.gather(labels, dim=-1, index=indices)

    # 4. decode box
    cx = indices % out_w + 0.5
    cy = indices // out_w + 0.5

    # box_offsets = box_offsets.flatten(start_dim=-2) * 16
    box_offsets = box_offsets.flatten(start_dim=-2)
    # box_offsets = box_offsets.clamp_min(0)

    # boxes are in output feature maps coordinates
    x1 = cx - torch.gather(box_offsets[:,0], dim=-1, index=indices)     # x1 = cx - left
    y1 = cy - torch.gather(box_offsets[:,1], dim=-1, index=indices)     # y1 = cy - top
    x2 = cx + torch.gather(box_offsets[:,2], dim=-1, index=indices)     # x2 = cx + right
    y2 = cy + torch.gather(box_offsets[:,3], dim=-1, index=indices)     # y2 = cy + bottom
    boxes = torch.stack((x1, y1, x2, y2), dim=-1)
    boxes[...,[0,2]] /= out_w                                           # normalize to [0,1]
    boxes[...,[1,3]] /= out_h

    return {
        "boxes": boxes,
        "scores": scores,
        "labels": labels
    }
