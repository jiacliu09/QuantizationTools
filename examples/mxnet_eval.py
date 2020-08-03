import mxnet
import numpy as np
import warnings
import scipy.spatial.distance
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', default="resnet50v1b-symbol.json")
    parser.add_argument('--params_file', default="resnet50v1b-0000.params")
    parser.add_argument('--cast_dtype', default="float32")
    parser.add_argument('--input', default="input.npy")
    parser.add_argument('--output', default="result.npy")
    parser.add_argument('--epoch', default=0, type=int)
    return parser.parse_args()

def distance(x, y):
    return scipy.spatial.distance.cosine(x.flatten(), y.flatten())

def load_var_node(name, shape):
    with open("params/%s" % name, 'r') as f:
        arr = f.read().strip().split('\n')[1]
        arr = arr.split(' ')
        arr = np.array([float(a) for a in arr]).reshape(shape).astype(np.float32)
    return arr

args = parse_args()

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    mx_model = mxnet.gluon.nn.SymbolBlock.imports(
            args.json_file,
            ['data'],
            args.params_file, ctx=mxnet.cpu())
    mx_model.cast(args.cast_dtype)
    # mx_model.export(args.prefix, args.epoch)
    # for name in mx_model.params.keys():
    #     param_data = mx_model.params[name].data().asnumpy()
    #     saved_data = load_var_node(name, param_data.shape)
    #     dist = distance(param_data, saved_data)
    #     print(name, dist)

    mx_inputs = mxnet.ndarray.from_numpy(np.load(args.input))
    mx_model.summary(mx_inputs)
    mx_outputs = mx_model(mx_inputs)
    for idx, mx_output in enumerate(mx_outputs):
        np.save(args.output, mx_output.asnumpy())
