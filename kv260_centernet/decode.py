import numpy as np
from scipy.ndimage.filters import maximum_filter


def maxpool(x: np.ndarray, kernel=3, fill=0.):
    return maximum_filter(x, size=kernel, mode='constant', cval=fill)

def decode(heatmap: np.ndarray, box_offsets: np.ndarray, num_detections=100):
    img_h, img_w, n_classes = heatmap.shape

    # pseudo-nms
    mask = heatmap == maxpool(heatmap)
    heatmap = heatmap * mask

    # get the best labels at each heatmap location
    heatmap = heatmap.reshape(-1, n_classes)    # flatten (H, W, C) -> (HW, C)
    labels = np.argmax(heatmap, axis=-1)        # (HW,)
    heatmap = np.max(heatmap, axis=-1)          # (HW,)

    # get topk. can also use heap here to have O(n) topk
    indices = np.argsort(-heatmap)[:num_detections]
    scores = heatmap[indices]
    labels = labels[indices]

    # decode boxes
    cx = indices % img_w + 0.5
    cy = indices // img_w + 0.5
    box_offsets = box_offsets * 16
    box_offsets = box_offsets.reshape(-1, 4)

    boxes = np.stack((cx,cy,cx,cy), axis=-1)
    boxes[...,:2] -= box_offsets[indices,:2]
    boxes[...,2:] += box_offsets[indices,2:]
    boxes *= 4
    
    return boxes, scores, labels
