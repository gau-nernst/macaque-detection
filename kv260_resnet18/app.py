import argparse
import time

import numpy as np
import cv2

import xir
import vart


def get_dpu(xmodel):
    g = xir.Graph.deserialize(xmodel)
    subgraphs = g.get_root_subgraph().toposort_child_subgraph()
    dpu_subgraph = subgraphs[1]

    dpu = vart.Runner.create_runner(dpu_subgraph, "run")
    return dpu


def prepare_dpu_buffers(dpu):
    # which step sets the batch size?
    input_tensors = dpu.get_input_tensors()         # list of tensors, since we can have multiple input/output tensors
    output_tensors = dpu.get_output_tensors()
    in_dims = input_tensors[0].dims                 # dim of first tensor
    out_dims = output_tensors[0].dims
    in_buffer = np.empty(in_dims, dtype=np.float32)
    out_buffer = np.empty(out_dims, dtype=np.float32)
    return in_buffer, out_buffer


def profile(args, runs=100):
    img = cv2.imread(args.image)
    img = img.astype(np.float32) / 255.

    g = xir.Graph.deserialize(args.xmodel)
    subgraphs = g.get_root_subgraph().toposort_child_subgraph()
    dpu_subgraph = subgraphs[1]

    dpu = vart.Runner.create_runner(dpu_subgraph, "run")
    # dpu = get_dpu(args.xmodel)      # this will fail. no idea why
    
    in_buffer, out_buffer = prepare_dpu_buffers(dpu)
    in_buffer[0] = img
    
    time0 = time.time()
    for _ in range(runs):
        job_id = dpu.execute_async([in_buffer], [out_buffer])
        dpu.wait(job_id)

    dtime = time.time() - time0
    print(f"{runs} runs")
    print(f"Network time: {dtime/runs*1000:.4f} ms/img")
    print(f"FPS: {runs/dtime:.4f} img/s")


def predict(args):
    img = cv2.imread(args.image)
    img = img.astype(np.float32) / 255.

    g = xir.Graph.deserialize(args.xmodel)
    subgraphs = g.get_root_subgraph().toposort_child_subgraph()
    dpu_subgraph = subgraphs[1]

    dpu = vart.Runner.create_runner(dpu_subgraph, "run")
    # dpu = get_dpu(args.xmodel)      # this will fail. no idea why

    in_buffer, out_buffer = prepare_dpu_buffers(dpu)
    print("input dim:", in_buffer.shape)
    print("output dim:", out_buffer.shape)

    in_buffer[0] = img      # copy image

    job_id = dpu.execute_async([in_buffer], [out_buffer])
    dpu.wait(job_id)

    preds = np.argmax(out_buffer[0])
    scores = np.exp(out_buffer[0])              # softmax
    scores = scores / scores.sum()
    print(f"Prediction: {preds}")
    print(f"Confidence: {scores[preds]*100:.2f}%")


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str)
    parser.add_argument("--image", type=str)
    parser.add_argument("--xmodel", type=str)
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
