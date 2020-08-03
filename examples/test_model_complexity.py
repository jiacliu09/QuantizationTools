import json
import numpy as np
import os, sys
import argparse

libpath = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(libpath)
import src

def summarize(results):
    nnz = sum([r[1] for r in results])
    # sparsity = [r[2] for r in results]
    total_p = sum([r[1] / (1-r[2]) for r in results])
    sparsity = 1 - nnz / total_p
    dense_flops = sum([r[3] for r in results])
    sparse_flops = sum([r[4] for r in results])
    return nnz, sparsity, dense_flops, sparse_flops

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--graph',
            default='models/sparse_resnet50v1b/IR_for_reconstruct_graph.json', help='full graph json')
    parser.add_argument('-p', '--params',
            default='models/sparse_resnet50v1b/IR_for_reconstruct_params.npz', help='full params path')

    parser.add_argument('-f', '--framework', choices=['mxnet', 'pytorch', 'tensorflow'],
            default='mxnet', help='model source framework')

    args = parser.parse_args()
    with open(args.graph, 'r') as f:
        json_graph = json.load(f)
    params = np.load(args.params, allow_pickle=True)['arr_0'][()]
    results  = src.compute_moffett_model_complexity(json_graph, params, args.framework)
    nnz, sparsity, dense_flops, sparse_flops = summarize(results)

    print('name\t\tnnz\t\tsparsity\t\tdense_flops\t\tsparse_flops\t\trate')
    for info in results:
        info.append(info[4] / sparse_flops * 100)
        line = '\t\t'.join(str(s)[:10] for s in info)
        print(line)

    print('nnz: {}, sparsity: {}, dense_flops: {}, sparse_flops: {}'.format(
        nnz, sparsity, dense_flops, sparse_flops))

