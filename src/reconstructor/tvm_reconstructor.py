import readline
import tvm
import tvm.relay as relay
import topi
import numpy as np
from tvm.contrib import graph_runtime

from .reconstructor import BaseReconstructor
from ..utils import *

def const(arr):
    if arr.dtype==np.float32:
        dtype = 'float32'
    elif arr.dtype == np.int32:
        dtype= 'int32'
    else:
        raise
    return relay.const(arr, dtype=dtype)

def get_input(node_dict, params, name):
    if name.isdigit():
        return node_dict[name]
    else:
        # return const(params[name])
        return relay.var(name, shape=params[name].shape)

class TVMReconstructor(BaseReconstructor):
    def __init__(self, graph, params, input_node_ids=[], output_node_ids=[], update_node_cfg=[]):
        super(TVMReconstructor, self).__init__(
                graph, params, input_node_ids, output_node_ids)
        self.gelu_param = {
                'gelu_a': np.float32(0.5),
                'gelu_b': np.float32(1.0),
                'gelu_c': np.sqrt(2 / np.pi),
                'gelu_d': np.float32(0.044715),
                'gelu_e': np.float32(3.0)
        }

    def _execute(self):
        self.node_dict = {}
        # self.node_dict['1'] = relay.const(np.zeros((1, 128)), dtype='int32')
        gelu_a = relay.var('gelu_a', shape=())
        gelu_b = relay.var('gelu_b', shape=())
        gelu_c = relay.var('gelu_c', shape=())
        gelu_d = relay.var('gelu_d', shape=())
        gelu_e = relay.var('gelu_e', shape=())

        self.node_dict['1'] = relay.var('input.1', shape=(1,128), dtype='int32')
        self.node_dict['2'] = relay.var('input.2', shape=(1,128), dtype='int32')
        for gnode in self.graph:
            name = gnode['name']
            op_type = gnode['op_type']
            attrs = gnode['attrs']
            del attrs['A_shape']
            del attrs['O_shape']

            inputs = gnode['inputs']

            if op_type == 'Const':
                arr = np.zeros(attrs['shape'], dtype=np.int32)
                y =  relay.const(arr, dtype='int32')

            elif op_type == 'expand_dims':
                x = get_input(self.node_dict, self.params, inputs[0])
                y = relay.expand_dims(x, attrs['axis'], attrs['num_newaxis'])

            elif op_type == 'reshape':
                x = get_input(self.node_dict, self.params, inputs[0])
                y = relay.reshape(x, attrs['newshape'])
            elif op_type == 'take':
                data = get_input(self.node_dict, self.params, inputs[0])
                indices = get_input(self.node_dict, self.params, inputs[1])
                y = relay.take(data, indices, axis=attrs['axis'][0], mode=attrs['mode'])
            elif op_type == 'one_hot':
                x = get_input(self.node_dict, self.params, inputs[0])
                cc1 = get_input(self.node_dict, self.params, inputs[1])
                cc2 = get_input(self.node_dict, self.params, inputs[2])
                y = relay.one_hot(x, cc1, cc2, **attrs)
            elif op_type == 'strided_slice':
                x = get_input(self.node_dict, self.params, inputs[0])
                y = relay.strided_slice(x, **attrs)
            elif op_type == 'mean':
                x = get_input(self.node_dict, self.params, inputs[0])
                y = relay.mean(x, axis=attrs['axis'],
                        exclude=attrs['exclude'],
                        keepdims=attrs['keepdims'])
            elif op_type == 'nn.dense':
                x = get_input(self.node_dict, self.params, inputs[0])
                weight = get_input(self.node_dict, self.params, inputs[1])
                y = relay.nn.dense(x, weight, units=attrs['units'][0])
            elif op_type == 'add':
                x1 = get_input(self.node_dict, self.params, inputs[0])
                x2 = get_input(self.node_dict, self.params, inputs[1])
                y = relay.add(x1, x2)
            elif op_type == 'subtract':
                x1 = get_input(self.node_dict, self.params, inputs[0])
                x2 = get_input(self.node_dict, self.params, inputs[1])
                y = relay.subtract(x1, x2)
            elif op_type == 'multiply':
                x1 = get_input(self.node_dict, self.params, inputs[0])
                x2 = get_input(self.node_dict, self.params, inputs[1])
                y = relay.multiply(x1, x2)
            elif op_type == 'power':
                x1 = get_input(self.node_dict, self.params, inputs[0])
                x2 = get_input(self.node_dict, self.params, inputs[1])
                y = relay.power(x1, x2)
            elif op_type == 'transpose':
                x = get_input(self.node_dict, self.params, inputs[0])
                y = relay.transpose(x, **attrs)
            elif op_type == 'tanh':
                x = get_input(self.node_dict, self.params, inputs[0])
                y = relay.tanh(x)
            elif op_type == 'squeeze':
                x = get_input(self.node_dict, self.params, inputs[0])
                y = relay.squeeze(x, **attrs)
            elif op_type == 'nn.batch_matmul':
                x1 = get_input(self.node_dict, self.params, inputs[0])
                x2 = get_input(self.node_dict, self.params, inputs[1])
                y = relay.nn.batch_matmul(x1, x2)
            elif op_type == 'nn.softmax':
                x = get_input(self.node_dict, self.params, inputs[0])
                y = relay.nn.softmax(x, **attrs)
            elif op_type == 'gelu':
                x = get_input(self.node_dict, self.params, inputs[0])
                y = x * gelu_a * (gelu_b + relay.tanh(
                           ( gelu_c * (x + gelu_d *
                                   relay.power(x, gelu_e)))))
            else:
                import pdb; pdb.set_trace()
                print( 'not supported op %s ' % op_type)
            self.node_dict[name] = y

        output_name = self.output_node_ids[0]
        output = self.node_dict[output_name]

        inputs = relay.analysis.free_vars(output)
        # inputs = [self.node_dict['1'], self.node_dict['2']]
        func = relay.Function(inputs, output)
        mod = tvm.IRModule()
        mod['main'] = func

        with relay.build_config(opt_level=0):
            graph, lib, params = relay.build(mod, 'llvm', params={})
        self.m = graph_runtime.create(graph, lib, tvm.cpu())

    def __call__(self, input_ids, segment_ids):
        self.m.set_input('input.1', tvm.nd.array(input_ids))
        self.m.set_input('input.2', tvm.nd.array(segment_ids))
        self.m.set_input(**self.params)
        self.m.set_input(**self.gelu_param)
        # import pdb; pdb.set_trace()
        self.m.run()
        rst = self.m.get_output(0).asnumpy()
        return rst
