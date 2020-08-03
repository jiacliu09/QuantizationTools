import tensorflow as tf
import numpy as np
import os

from .reconstructor import BaseReconstructor
from ..utils.tf_functions import function_factory
from ..postprocess import postprocessor_factory

spec = lambda x, i : x + ':' + str(i)

class TFReconstructor(BaseReconstructor):
    def __init__(self, graph, params, input_node_ids=[], output_node_ids=[], update_node_cfg=[], postprocessor=None):
        super(TFReconstructor, self).__init__(
                graph, params, input_node_ids, output_node_ids, update_node_cfg)
        self.real_input_shapes = self._transpose_data_shape()
        self.set_postprocessor(postprocessor)

        self._tf_graph = tf.get_default_graph()
        self._sess = tf.Session(graph=self._tf_graph)
        self._execute_done = False

    def _execute(self, reset=False, input_dict=None):
        if self._execute_done and not reset:
            return
        with self._tf_graph.as_default():
            self.node_dict = self._get_node_dict()
            if input_dict:
                for kk in input_dict.keys():
                    if kk in self.node_dict:
                        self.node_dict[kk] = input_dict[kk]
            self._parse_model()
            if self._postprocessor is not None:
                heatmaps = [self.node_dict[spec(name, 0)] for name in self.output_node_ids]
                output = self._postprocessor(*heatmaps)
                self.output_node_ids = ['post']
                self.node_dict['post:0'] = output
        self._execute_done = True

    def _transpose_data_shape(self):
        transpose_input_shapes = []
        for shape in self.input_node_shapes:
            if len(shape) == 4:
                b, c, h, w = shape
                transpose_input_shapes.append((-1, h, w, c))
            else:
                transpose_input_shapes.append(shape)
        return transpose_input_shapes

    def set_postprocessor(self, postprocessor):
        if isinstance(postprocessor, str):
            self._postprocessor = postprocessor_factory[postprocessor]
        else:
            self._postprocessor = postprocessor

    def _get_node_dict(self):
        node_dict = {}
        for name in self.params:
            arr = tf.constant(self.params[name], dtype=tf.float32, name=name)
            node_dict[spec(name, 0)] = arr
        for name, shape in zip(self.input_node_ids, self.real_input_shapes):
            shape = (s if s >= 1 else None for s in shape)
            node_dict[spec(name, 0)] = tf.placeholder(tf.float32, shape=shape, name=name)
        return node_dict


    def _parse_model(self):

        for gnode in self.graph:
            attrs = gnode.get('attrs')
            if attrs is None:  attrs = {}
            inputs = gnode.get('inputs')
            if inputs is None:  inputs = []
            inputs = [self.node_dict[spec(in_, 0)] for in_ in inputs]
            name = gnode.get('name')
            x = function_factory[gnode['op_type']](inputs, name, **attrs)
            if isinstance(x, tf.Tensor):
                x = [x]

            # if gnode['op_type'] == 'split':
            #     assert len(x) > 1
            #     for idx, item in enumerate(x):
            #         self.node_dict[spec(name, idx)] = item
            # else:
            for idx, tensor in enumerate(x):
                self.node_dict[spec(name, idx)] = tensor

    @property
    def tf_graph(self):
        return self._tf_graph

    def __call__(self, *args):
        self._execute()
        input_tsrs = [self.node_dict[spec(name, 0)] for name in self.input_node_ids]
        assert len(args) == len(input_tsrs)
        feed_dict = dict((t, v) for (t, v) in zip(input_tsrs, args))
        output_tsrs = [self.node_dict[spec(name, 0)] for name in self.output_node_ids]

        output = self._sess.run(output_tsrs, feed_dict=feed_dict)

        if len(output) > 1:
            return dict((spec(name,0), tsr) for (name, tsr) in zip(self.output_node_ids, output))
        else:
            return output[0]

    def save_graph(self, path):
        self._execute()
        tf.io.write_graph(self.tf_graph, os.path.dirname(path), os.path.basename(path), as_text=False)


class TFReconstructorTrain(TFReconstructor):
    def __init__(self, graph, params, input_node_ids=[], output_node_ids=[], update_node_cfg=[], is_training=False, postprocessor=None):
        super(TFReconstructorTrain, self).__init__(
                graph, params, input_node_ids, output_node_ids, update_node_cfg, postprocessor)
        self._is_training = is_training


    def _get_node_dict(self):
        def _get_maks(npr):
            if npr.ndim != 4:
                return None
            index = np.nonzero(npr)
            mask = np.zeros_like(npr)
            mask[index] = 1.0
            return mask

        node_dict = {}
        for name in self.params:
            arr = self.params[name]
            mask = _get_maks(arr)
            node_dict[name] = (arr, mask)

        for name, shape in zip(self.input_node_ids, self.real_input_shapes):
            shape = (s if s >= 0 else None for s in shape)
            node_dict[name] = tf.placeholder(tf.float32, shape=shape, name=name)
        return node_dict

    def _parse_model(self):
        # is_training = tf.placeholder_with_default(self._is_training, (), "is_training", dtype=tf.boolean)
        for gnode in self.graph:
            attrs = gnode.get('attrs')
            if attrs is None:  attrs = {}
            inputs = gnode.get('inputs')
            if inputs is None:  inputs = []
            inputs = [self.node_dict[in_] for in_ in inputs]
            name = gnode.get('name')
            op_type = gnode['op_type']
            if op_type in ['nn.conv2d', 'nn.batch_norm', 'nn.dense', 'nn.bias_add']:
                op_type += '_train'
            # if op_type == 'nn.bias_add_train':
            #     import pdb; pdb.set_trace()
            x = function_factory[op_type](inputs, name, is_training=self._is_training, **attrs)
            self.node_dict[name] = x

    def frozen(self):
        self._execute()
        sess = tf.Session(graph=self._tf_graph)
        sess.run(tf.global_variables_initializer())
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, sess.graph_def, self.output_node_ids)
        print('Done')
        return frozen_graph_def

    def model(self, input_dict=None, is_training=True):
        self._is_training = is_training
        self._execute(reset=True, input_dict=input_dict)
        return dict((k, self.node_dict[k]) for k in self.output_node_ids)
