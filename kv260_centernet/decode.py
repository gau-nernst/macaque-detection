import numpy as np
# from scipy.ndimage.filters import maximum_filter


def maxpool(a: np.ndarray, kernel=3):
    # return maximum_filter(a, size=kernel, mode='constant', cval=0)
    h, w, c = a.shape
    pad = (kernel-1) // 2
    a = a.copy()
    padded_a = np.zeros((h+pad*2, w+pad*2, c), dtype=a.dtype)
    padded_a[pad:-pad,pad:-pad,:] = a
    for i in range(c):
        for i_x in range(pad, w+pad):
            for i_y in range(pad, h+pad):
                a[i_y-pad,i_x-pad,i] = np.max(padded_a[i_y-pad:i_y+pad+1,i_x-pad:i_x+pad+1,i])

    return a

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
