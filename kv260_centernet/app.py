import argparse
import time

import numpy as np
import cv2

import xir
import vart

from decode import decode


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
    print(f'Detect {len(boxes)} macaques')

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

if __name__ == "__main__":
    args = get_args_parser().parse_args()

    if args.command == "predict":
        predict(args)
    elif args.command == "profile":
        profile(args)
    else:
        raise ValueError()
