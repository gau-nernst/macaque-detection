import numpy as np
# from scipy.ndimage.filters import maximum_filter
# from numba import jit

# @jit(nopython=True, parallel=True)
def maxpool(a: np.ndarray, kernel=3):
    # return maximum_filter(a, size=kernel, mode='constant', cval=0)
    h, w, c = a.shape
    pad = (kernel-1) // 2
    a = a.copy()
    padded_a = np.zeros((h+pad*2, w+pad*2, c), dtype=a.dtype)
    padded_a[pad:-pad,pad:-pad,:] = a
    for i_y in range(h):
        for i_x in range(w):
            np.max(padded_a[i_y:i_y+kernel,i_x:i_x+kernel], axis=(0,1), out=a[i_y,i_x])
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

# @jit(nopython=True, parallel=True)
def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float):
    indices = np.argsort(-scores)
    areas = (boxes[...,2] - boxes[...,0]) * (boxes[...,3] - boxes[...,1])
    
    num_boxes = boxes.shape[0]
    keep = np.zeros(num_boxes, dtype=np.uint8)
    remove = np.zeros(num_boxes, dtype=np.uint8)
    for i, idx in enumerate(indices):
        if remove[idx]:
            continue
        keep[idx] = 1
        
        for idx2 in indices[i+1:]:
            if remove[idx2]:
                continue
            ix1 = max(boxes[idx,0], boxes[idx2,0])
            iy1 = max(boxes[idx,1], boxes[idx2,1])
            ix2 = min(boxes[idx,2], boxes[idx2,2])
            iy2 = min(boxes[idx,3], boxes[idx2,3])

            inter = max(0, ix2-ix1) * max(0, iy2-iy1)
            iou = inter / (areas[idx] + areas[idx2] - inter)
            if iou > threshold:
                remove[idx2] = 1

    return np.where(keep)[0]
