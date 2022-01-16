import argparse

import numpy as np
import cv2

import xir
import vart

def app(args):
    img = cv2.imread(args.image)
    img = img.astype(np.float32) / 255.


    g = xir.Graph.deserialize(args.xmodel)
    subgraphs = g.get_root_subgraph().toposort_child_subgraph()
    dpu_subgraph = subgraphs[1]

    dpu = vart.Runner.create_runner(dpu_subgraph, "run")

    input_tensors = dpu.get_input_tensors()
    in_dims = input_tensors[0].dims
    print("input dims", in_dims)
    batch_size = in_dims[0]                     # which step sets the batch size?

    output_tensors = dpu.get_output_tensors()
    out_dims = output_tensors[0].dims
    print("output dims", out_dims)

    in_buffer = np.empty(in_dims, dtype=np.float32)
    out_buffer = np.empty(out_dims, dtype=np.float32)

    in_buffer[0] = img

    job_id = dpu.execute_async([in_buffer], [out_buffer])
    dpu.wait(job_id)

    preds = np.argmax(out_buffer[0])
    scores = np.exp(preds)              # softmax
    scores = scores / scores.sum()
    print(f"Prediction: {preds}")
    print(f"Confidence: {scores[preds]*100:.2f}%")


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="./image.jpg")
    parser.add_argument("--xmodel", type=str, default="./model.xmodel")
    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    app(args)
