#!/usr/bin/python3
from typing import Tuple
import time

import numpy as np
import cv2
import telebot

# Vitis AI libraries
import xir
import vart


API_KEY = "ENTER BOT TOKEN"
CHANNEL_ID = "CH_ID"
DEVICE_NAME = "LOCATION"
THRESHOLD = 0.3
XMODEL_PATH = "centernet.xmodel"

BOT = telebot.TeleBot(API_KEY)
CAM = cv2.VideoCapture(0)


@BOT.message_handler(commands=["Help"])
def help_Message(message):
    out_message= "Hi there, I am a Maccaque Detection bot"
    BOT.send_message(message.chat.id, out_message)

def call_Debug(message):
    return message.text.lower() == 'debug'

@BOT.message_handler(func=call_Debug)
def debug_Message(message):
    out_message = f"Debugging {DEVICE_NAME}."
    BOT.send_message(CHANNEL_ID, out_message)
    BOT.send_message(message.chat.id, out_message)


# operations
def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))

def maxpool(a: np.ndarray, kernel=3):
    # return maximum_filter(a, size=kernel, mode='constant', cval=0)
    h, w = a.shape
    pad = (kernel-1) // 2
    a = a.copy()
    padded_a = np.zeros((h+pad*2, w+pad*2), dtype=a.dtype)
    padded_a[pad:-pad,pad:-pad,:] = a
    for i_y in range(h):
        for i_x in range(w):
            np.max(padded_a[i_y:i_y+kernel,i_x:i_x+kernel], axis=(0,1), out=a[i_y,i_x])
    return a


# DPU helpers
def initialize_dpu(xmodel_path):
    g = xir.Graph.deserialize(xmodel_path)
    subgraphs = g.get_root_subgraph().toposort_child_subgraph()
    dpu_subgraph = subgraphs[1]

    dpu = vart.Runner.create_runner(dpu_subgraph, "run")
    return dpu

def prepare_dpu_buffers(dpu):
    input_tensors = dpu.get_input_tensors()
    output_tensors = dpu.get_output_tensors()
    in_buffers = [np.empty(x.dims, dtype=np.float32) for x in input_tensors]
    out_buffers = [np.empty(x.dims, dtype=np.float32) for x in output_tensors]
    return in_buffers, out_buffers


# detection helpers
def decode_detections(heatmap: np.ndarray, box_offsets: np.ndarray, image_shape: Tuple[int, int], num_detections=100):
    output_h, output_w = heatmap.shape[:2]
    mask = heatmap == maxpool(heatmap)          # pseudo-nms
    heatmap = heatmap * mask
    heatmap = np.flatten(heatmap)               # (H, W, 1) -> (HW,)
    
    indices = np.argsort(-heatmap)[:num_detections]     # get topk
    scores = heatmap[indices]
    
    # decode boxes
    cx = indices % output_w + 0.5
    cy = indices // output_w + 0.5
    box_offsets = box_offsets.reshape(-1, 4)    # (H, W, 4) -> (HW, 4)

    boxes = np.stack((cx,cy,cx,cy), axis=-1)
    boxes[...,:2] -= box_offsets[indices,:2]
    boxes[...,2:] += box_offsets[indices,2:]

    boxes[...,:2] *= image_shape[1] / output_w
    boxes[...,2:] *= image_shape[0] / output_h

    return boxes, scores

def draw_boxes(img: np.ndarray, boxes: np.ndarray):
    img = img.copy()
    boxes = boxes.round.astype(int)
    for box in boxes:
        cv2.rectangle(img, box[:2], box[2:], (255,0,0), 2)
    return img


def predict(img: np.ndarray, dpu):
    '''Input is an uint8 RGB image
    '''
    img_input = img.astype(np.float32) / 255.       # [0,255] uint8 -> float32 [0,1]

    in_buffers, out_buffers = prepare_dpu_buffers(dpu)
    img_input = cv2.resize(img_input, in_buffers[0][0].shape[:2])
    in_buffers[0][0] = img_input

    job_id = dpu.execute_async(in_buffers, out_buffers)
    dpu.wait(job_id)

    heatmap, box_offsets = out_buffers
    boxes, scores = decode_detections(sigmoid(heatmap[0]), box_offsets[0], img_input.shape)
    mask = scores > THRESHOLD
    boxes = boxes[mask]
    scores = scores[mask]
    
    return boxes, scores

# BOT.polling()

if __name__ == "__main__":
    state = 0
    while True:
        # get image
        ret, img = CAM.read()
        if ret:
            boxes, scores = predict(img)
        
            if len(boxes) > 0:
                if state == 0:
                    BOT.send_message(CHANNEL_ID, "yo")
                    state = 1

                    # draw bounding box
                    img = draw_boxes(img, boxes)
                    ret, img_bin = cv2.imencode(".jpg", img)
                    BOT.send_photo(CHANNEL_ID, img_bin)

                time.sleep(10)
            
            else:
                state = 0
