import argparse
import os

import numpy as np
import torch
import cv2
from tqdm import tqdm

from model import CenterNet, decode_detections
from visualize import get_model, draw_boxes, imagenet_mean, imagenet_std

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

imagenet_mean = imagenet_mean.astype(np.float32).reshape(1,1,3)
imagenet_std = imagenet_std.astype(np.float32).reshape(1,1,3)


@torch.no_grad()
def inference(model: CenterNet, image: np.ndarray, input_size=(512,512), threshold=0.3):
    image = cv2.resize(image, input_size).astype(np.float32) / 255              # resize and uint8 -> float32
    image = (image - imagenet_mean) / imagenet_std                              # normalize
    image = torch.from_numpy(image.transpose(2,0,1)).unsqueeze(0)               # HWC to NCHW

    heatmap, box_offsets = model(image.to(DEVICE))
    detections = decode_detections(heatmap.sigmoid(), box_offsets)
    detections = {k: v.squeeze().cpu().numpy() for k, v in detections.items()}

    mask = detections["scores"] > threshold                             # filter
    boxes = detections["boxes"][mask]
    scores = detections["scores"][mask]

    return boxes, scores


def get_args_parser():
    parser = argparse.ArgumentParser()    
    parser.add_argument("input")
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--weights")
    parser.add_argument("--output_dir", default="centernet")

    parser.add_argument("--img_w", type=int, default=512)
    parser.add_argument("--img_h", type=int, default=512)

    parser.add_argument("--quant_model", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.3)

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()

    input_size = (args.img_w, args.img_h)
    model = get_model(args)
    if args.input.endswith((".mp4", ".mov")):
        video = cv2.VideoCapture(args.input)
        if not video.isOpened():
            raise RuntimeError("Failed to open the video")

        vid_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)

        out_name = os.path.join(args.output_dir, "output3.avi")
        codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        video_out = cv2.VideoWriter(out_name, codec, fps, (vid_w, vid_h))

        with tqdm(total=vid_len) as pbar:
            while True:
                ret, frame = video.read()
                if ret:
                    boxes, scores = inference(model, frame, input_size=input_size, threshold=args.threshold)
                    boxes[...,[0,2]] *= vid_w
                    boxes[...,[1,3]] *= vid_h

                    texts = [f"{x*100:.0f}%" for x in scores]
                    frame = draw_boxes(frame, boxes, texts)
                    video_out.write(frame)
                    pbar.set_description(f"{len(boxes)} detections")
                    pbar.update(1)
                else:
                    break
