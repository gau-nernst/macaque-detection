import argparse
import time

import numpy as np
import cv2

import xir
import vart

from decode import decode

import telebot
import requests

API_KEY = "ENTER BOT TOKEN"
CHANNEL_ID = "CH_ID"
DEVICE_NAME = "LOCATION"

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
    # Following command used to send message directly to channel
    requests.post(f'https://api.telegram.org/bot{API_KEY}/sendMessage?chat_id={CHANNEL_ID}&text={out_message}')
    BOT.send_message(message.chat.id, out_message)

def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))


def prepare_dpu_buffers(dpu):
    input_tensors = dpu.get_input_tensors()
    output_tensors = dpu.get_output_tensors()
    in_buffers = [np.empty(x.dims, dtype=np.float32) for x in input_tensors]
    out_buffers = [np.empty(x.dims, dtype=np.float32) for x in output_tensors]
    return in_buffers, out_buffers


def profile(args, runs=100):
    img = cv2.imread(args.image)
    img = img.astype(np.float32) / 255.

    g = xir.Graph.deserialize(args.xmodel)
    subgraphs = g.get_root_subgraph().toposort_child_subgraph()
    dpu_subgraph = subgraphs[1]

    dpu = vart.Runner.create_runner(dpu_subgraph, "run")
    
    in_buffers, out_buffers = prepare_dpu_buffers(dpu)
    img = cv2.resize(img, in_buffers[0][0].shape[:2])
    in_buffers[0][0] = img
    
    time0 = time.time()
    for _ in range(runs):
        job_id = dpu.execute_async(in_buffers, out_buffers)
        dpu.wait(job_id)

    dtime = time.time() - time0
    print(f"{runs} runs")
    print(f"Network time: {dtime/runs*1000:.4f} ms/img")
    print(f"FPS: {runs/dtime:.4f} img/s")


def predict(args):
    img = cv2.imread(args.image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_input = img.astype(np.float32) / 255.

    g = xir.Graph.deserialize(args.xmodel)
    subgraphs = g.get_root_subgraph().toposort_child_subgraph()
    dpu_subgraph = subgraphs[1]

    dpu = vart.Runner.create_runner(dpu_subgraph, "run")

    in_buffers, out_buffers = prepare_dpu_buffers(dpu)
    img_input = cv2.resize(img_input, in_buffers[0][0].shape[:2])
    in_buffers[0][0] = img_input

    job_id = dpu.execute_async(in_buffers, out_buffers)
    dpu.wait(job_id)

    heatmap, box_offsets = out_buffers
    boxes, scores, labels = decode(sigmoid(heatmap[0]), box_offsets[0])
    mask = scores > args.threshold
    boxes = boxes[mask]
    scores = scores[mask]
    broadcast_msg = f'Detect {len(boxes)} macaques at {DEVICE_NAME}'
    print(broadcast_msg)
    if len(boxes)>0:
            requests.post(f'https://api.telegram.org/bot{API_KEY}/sendMessage?chat_id={CHANNEL_ID}&text={broadcast_msg}')


    boxes[...,[0,2]] *= img.shape[1] / img_input.shape[1]
    boxes[...,[1,3]] *= img.shape[0] / img_input.shape[0]
    boxes = boxes.round().astype(int)
    for x1,y1,x2,y2 in boxes:
        cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 2)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('prediction.jpg', img)


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("command")
    parser.add_argument("--image")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--xmodel")
    parser.add_argument("--runs", type=int, default=100)
    return parser
BOT.polling()

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
                    img_bin = cv2.imencode(".jpg", img)[1]
                    BOT.send_photo(CHANNEL_ID, img_bin)

                time.sleep(10)
            
            else:
                state = 0
