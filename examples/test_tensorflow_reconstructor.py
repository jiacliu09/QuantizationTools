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
    parser.add_argument('-s', '--save_path', default='pbs/resnet50v1b_sp.pb', help='save pb path')
    parser.add_argument('--post', default='', help='postprocess key')
    parser.add_argument('--shape', default=224, help='', type=int)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    with open(args.graph, 'r') as f:
        graph = json.load(f)
    params = np.load(args.params, allow_pickle=True)['arr_0'][()]
    params = src.transform_weight_from_mxnet_to_tensorflow(params)
    trc = src.TFReconstructor(graph, params)
    # import pdb; pdb.set_trace()
    if args.post:
        trc.set_postprocessor(args.post)

    if args.input_npy:
        x = np.load(args.input_npy)
    else:
        shape = [1, 3, args.shape, args.shape]
        x = np.random.uniform(0, 1, shape)
    x = x.transpose((0,2,3,1))
    y = trc(x)
    if args.result_npy and args.input_npy:
        result = np.load(args.result_npy)
        print('distance', src.distance(y, result))

    if args.save_path:
        trc.save_graph(args.save_path)
        print('*** reconstruct pb saved at {} ***'.format(args.save_path))
