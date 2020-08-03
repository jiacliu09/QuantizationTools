import tensorflow as tf
import numpy as np
import glog
import os, json
from ..utils.tf_functions import *
from ..utils import *
from ..reconstructor import BaseReconstructor

spec = lambda x, i : x + ':' + str(i)
despec = lambda x: x.split(':')[0]

class BaseQuan(BaseReconstructor):
    Need_quantized_function = [
            mir_conv2d_bias_relu, mir_conv2d_bias, conv2d,
            # mir_scale_bias, mir_scale_bias_relu,
            ]

    Not_Need_requantize_op = ['Const', 'nn.bias_add']

    def __init__(self, weight_quan, image_path, graph, params, table=None):
        super(BaseQuan, self).__init__(graph, params, strip_inputs=False)
        self.weight_quan = weight_quan
        # self.params = params
        # self.graph = graph
        self.image_list = get_image_list(image_path)
        self.info_list = []
        self.act_list = []
        self.table = self._get_calib_table(table)
        self.node_dict = self._get_node_dict()
        self._set_requantize_flag()

    def _set_requantize_flag(self):
        # not_use_v2_node = ['280']
        for gnode in self.graph:
            if gnode['op_type'] in self.Not_Need_requantize_op:
                gnode['attrs']['quantize'] = 0
            else:
                gnode['attrs']['quantize'] = 1

            # if gnode['name'] in not_use_v2_node:
            gnode['attrs']['use_v2'] = 1
            # else:
            #     gnode['attrs']['use_v2'] = 1

            input_requantize = []
            inputs =  get_input_node(gnode)
            for in_ in inputs:
                node = find_node(self.graph, in_)
                input_requantize.append(node['attrs'].get('quantize', 0))

            gnode['attrs']['input_requantize'] = input_requantize

    def _get_calib_table(self, table):
        if table:
            if isinstance(table, str):
                with open(table, 'r') as f:
                    table = json.load(f)
        return table

    def _get_node_dict(self):
        node_dict = {}
        for name in self.params:
            arr = tf.constant(self.params[name], dtype=tf.float32, name=name)
            node_dict[spec(name,0)] = arr
        return node_dict

    def _prepare_feed_dict(self, function, inputs, name, **kwargs):
        if function.__name__ == 'const':
            images = process_image_batch(self.image_list[:25])
            # images = preprocess_on_coco(self.image_list[:25])
            # images = load_cifar10(self.image_list[:25])
            feed_dict = {spec(name, 0): images}
        else:
            # import pdb; pdb.set_trace()
            feed_dict = dict((k, self.params[despec(k.name)]) for k in inputs)
        return feed_dict

    def execute(self):

        for gnode in self.graph:
            attrs = gnode.get('attrs')
            if attrs is None:  attrs = {}
            inputs = gnode.get('inputs')
            if inputs is None:  inputs = []
            # inputs = [self.node_dict[spec(in_,0)] for in_ in inputs]
            name = gnode.get('name')
            x = self.inference(gnode['op_type'])(inputs, name, **attrs)
            if isinstance(x, tf.Tensor):
                x = [x]
            for idx, tensor in enumerate(x):
                self.node_dict[spec(name,0)] = tensor

    def inference(self, function):
        raise NotImplementedError
