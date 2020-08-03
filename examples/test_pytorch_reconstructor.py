import json
import torch
import numpy as np
import argparse
import sys, os

libpath = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(libpath)
import src

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--graph', default='./models/sparse_resnet50v1b/IR_for_reconstruct_graph.json', help='graph json')
    parser.add_argument('-p', '--params', default='./models/sparse_resnet50v1b/IR_for_reconstruct_params.npz', help='params file')
    parser.add_argument('-i', '--input_npy', default=None, help='input numpy or none, when input is none, random input')
    parser.add_argument('-o', '--result_npy', default=None, help='result numpy to compare reconstructor result')
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    with open(args.graph, 'r') as f:
        graph = json.load(f)

    params = np.load(args.params, allow_pickle=True)['arr_0'][()]
    trc = src.TorchReconstructor(graph, params)
    trc.load_weights()
    trc.eval()
    print(trc)
    print("spec_name, nnz, sparsity")
    for module in trc.model:
        print(module.spec_name, module.nnz, module.sparsity)
    if args.input_npy:
        x = torch.from_numpy(np.load(args.input_npy))
    else:
        x = torch.randn((1,3,224,224))
    if args.result_npy and args.input_npy:
        result = np.load(args.result_npy)
        y = trc(x)
        print('distance', src.distance(y.detach().numpy(), result))
