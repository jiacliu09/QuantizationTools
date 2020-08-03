import numpy as np
import tvm
import tvm.relay as relay
from collections import namedtuple

from .utils import cast_params

def from_mxnet(model_path, epoch, img_shape):
    import mxnet as mx
    import json
    batch_size = 1
    img_shape = (3,) + tuple(img_shape)

    mx_sym, arg_params, aux_params = mx.model.load_checkpoint(model_path, epoch=epoch)
    arg_params = cast_params(arg_params)
    aux_params = cast_params(aux_params)
    graph_inputs = [graph_input for graph_input in mx_sym.list_inputs()
                      if graph_input not in arg_params and graph_input not in aux_params]
    input_shape = (batch_size, ) + img_shape
    shape_dict = {}
    for input_name in graph_inputs:
        shape_dict[input_name] = input_shape
    mod, params = relay.frontend.from_mxnet(mx_sym, shape_dict, dtype='float32', arg_params=arg_params, aux_params=aux_params)
    Model = namedtuple('Model', ['mod', 'params', 'graph_inputs'])
    model = Model(mod, params, graph_inputs)
    return model

def from_tensorflow(model_path, img_shape):
    import tensorflow as tf
    import tvm.relay.testing.tf as tf_testing

    def analyze_tf_inputs_outputs(graph):
        ops = graph.get_operations()
        outputs_set = set(ops)
        inputs = []
        for op in ops:
            if len(op.inputs) == 0 and op.type != 'Const':
                inputs.append(op)
            else:
                for input_tensor in op.inputs:
                    if input_tensor.op in outputs_set:
                        outputs_set.remove(input_tensor.op)
        outputs = [op.name for op in outputs_set]
        outputs.sort()
        inputs = [op.name for op in inputs]
        return inputs, outputs

    shape_dict = {}
    with tf.compat.v1.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        graph = tf.import_graph_def(graph_def, name='')
        graph_inputs, graph_outputs = analyze_tf_inputs_outputs(tf.get_default_graph())
        for node in graph_def.node:
            if node.name in graph_inputs:
                get_shape = ()
                input_lenth = len(node.attr['shape'].shape.dim)
                if input_lenth:
                    for i in range(input_lenth):
                        dim_size = node.attr['shape'].shape.dim[i].size
                        dim_size = dim_size if dim_size != -1 else None
                        get_shape += (dim_size,)
                if len(get_shape) == 0 or not all(get_shape[1:]):
                    input_shape = (1,) + tuple(img_shape) + (3,)
                elif not all(get_shape) and all(get_shape[1:]):
                    input_shape = (1,) + get_shape[1:]
                else:
                    input_shape = get_shape
                shape_dict[node.name] = input_shape
        # Call the utility to import the graph definition into default graph.
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)
        # Add shapes to the graph.
        with tf.compat.v1.Session() as sess:
            graph_def = tf_testing.AddShapesToGraphDef(sess, graph_outputs)

    mod, params = relay.frontend.from_tensorflow(graph_def, layout=None, shape=shape_dict)
    Model = namedtuple('Model', ['mod', 'params', 'graph_inputs'])
    model = Model(mod, params, graph_inputs)
    return model

def from_onnx(model_path, img_shape):
    import onnx

    onnx_model = onnx.load(model_path)
    shape_dict = {}
    graph_inputs = []
    for item in onnx_model.graph.input:
        input_name = item.name
        get_shape = ()
        for i in range(len(item.type.tensor_type.shape.dim)):
            get_shape += (item.type.tensor_type.shape.dim[i].dim_value, )
        if all(get_shape):
            input_shape = get_shape
        else:
            input_shape = (1, 3,) + tuple(img_shape)
        shape_dict[input_name] = input_shape
        graph_inputs.append(input_name)

    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    Model = namedtuple('Model', ['mod', 'params', 'graph_inputs'])
    model = Model(mod, params, graph_inputs)
    return model

def gen_relay_ir(platform, model_path, img_shape, epoch = 0):
    if platform == 'mxnet':
        return from_mxnet(model_path, epoch, img_shape)
    elif platform == 'tensorflow':
        return from_tensorflow(model_path, img_shape)
    elif platform == 'onnx':
        return from_onnx(model_path, img_shape)
    else:
        raise RuntimeError(
                'Unknown platform type: {}'.format(platform))

