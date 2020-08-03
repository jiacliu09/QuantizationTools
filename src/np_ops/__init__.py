import numpy as np
from .layer import fc, conv

def compute_conv_flops(weight, input_shape, stride, padding):
    fake_inputs = np.ones(input_shape, dtype=np.float32)
    fake_dense_weight = np.ones_like(weight)
    dense_flops = conv(fake_inputs, fake_dense_weight, stride, padding).sum()
    nonzero_index = np.nonzero(weight)
    fake_sparse_weight = np.zeros_like(weight)
    fake_sparse_weight[nonzero_index] = 1.0
    sparse_flops = conv(fake_inputs, fake_sparse_weight, stride, padding).sum()
    return dense_flops, sparse_flops

def compute_dense_flops(weight, input_shape):
    fake_inputs = np.ones(input_shape, dtype=np.float32)
    fake_dense_weight = np.ones_like(weight)
    dense_flops = fc(fake_inputs, fake_dense_weight).sum()
    nonzero_index = np.nonzero(weight)
    fake_sparse_weight = np.zeros_like(weight)
    fake_sparse_weight[nonzero_index] = 1.0
    sparse_flops = fc(fake_inputs, fake_sparse_weight).sum()
    return dense_flops, sparse_flops

def compute_moffett_model_complexity(graph, params, framework='pytorch'):
    results = []
    for node in graph:
        if node['op_type'] not in ['nn.conv2d', 'nn.dense', 'nn.conv2d_bias', 'nn.conv2d_bias_relu']:
            continue
        weight = params[node['inputs'][1]]
        nnz = np.nonzero(weight)[0].size
        sparsity = 1 - nnz * 1. / weight.size
        name = node['name'] + '_' + node['op_type'].lstrip('nn.')
        input_shape = node['attrs']['A_shape'][0]
        if node['op_type'] == 'nn.conv2d':
            if framework in ['tensorflow']:
                weight = weight.transpose([3,2,0,1])
                b, h, w, c = input_shape
                input_shape = [b, c, h, w]
            dense_flops, sparse_flops = compute_conv_flops(weight, input_shape,
                    stride=node['attrs']['strides'][0],
                    padding=node['attrs']['padding'][0])
        elif node['op_type'] == 'nn.dense':
            if framework in ['pytorch', 'mxnet']:
                weight = weight.transpose((1,0))
            dense_flops, sparse_flops = compute_dense_flops(weight, input_shape)
        item = [name, nnz, sparsity, dense_flops, sparse_flops]
        results.append(item)
    return results
