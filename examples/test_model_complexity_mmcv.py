import json
import torch
from mmcv.cnn import get_model_complexity_info

import numpy as np
import argparse
import os, sys
libpath = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(libpath)
import src

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--graph', default='./models/sparse_resnet50v1b/IR_for_reconstruct_graph.json', help='graph json')
    parser.add_argument('--shape', type=int, nargs=3, default=[3,224,224], help='input shape')
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    with open(args.graph, 'r') as f:
        graph = json.load(f)
    trc = src.TorchReconstructor(graph)
    trc.eval()

    flops, params = get_model_complexity_info(trc, tuple(args.shape))
