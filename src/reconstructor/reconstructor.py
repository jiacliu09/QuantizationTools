import torch
import numpy as np
import torch.nn as nn
from ..utils import find_node, get_input_node


def find_unused_node(node_infos, in_ids, out_ids, strip_inputs=True):
    all_ids = [node['name'] for node in node_infos]
    flags = dict((id_, False) for id_ in all_ids)
    left_ids = set(out_ids)
    while(left_ids) :
        for id_ in left_ids:
            flags[id_] = True
        needed_ids = []
        for id_ in left_ids:
            node = find_node(node_infos, id_)
            needed_ids += get_input_node(node)
        needed_ids = [id_ for id_ in needed_ids if id_ not in in_ids]
        left_ids = set(needed_ids)

    if not strip_inputs:
        for id_ in in_ids:
            flags[id_] = True
    return [id_ for id_ in all_ids if not flags[id_]]

def update_cfg(nodes, node_cfg):
    node = find_node(nodes, node_cfg['name'])
    node.update(node_cfg)

class BaseReconstructor(object):
    # supported_ops = ['nn.conv2d', 'nn.relu', 'nn.batch_norm',
    #         'nn.global_avg_pool2d', 'nn.batch_flatten', 'nn.dense',
    #         'nn.max_pool2d', 'add', 'nn.avg_pool2d', 'nn.bias_add',
    #         'nn.conv2d_transpose', 'sigmoid', 'multiply', 'divide',
    #         'subtract', 'nn.pad', 'clip']

    def __init__(self, graph, params, input_node_ids=[], output_node_ids=[], update_node_cfg=[], strip_inputs=True):
        super(BaseReconstructor, self).__init__()
        if update_node_cfg:
            for node_cfg in update_node_cfg:
                update_cfg(graph, node_cfg)

        self.input_node_ids = input_node_ids
        self.input_node_shapes = []
        self.output_node_ids = output_node_ids
        self.strip_inputs = strip_inputs
        self.graph = self._find_input_output_nodes(graph)
        if params is not None:
            self.params = self._strip_unused_params(params)
        else:
            self.params = {}

    def get_node_inputs(self, name):
        node = find_node(self.graph, name)
        if node is None: return []
        inputs = node['inputs']
        if inputs is None: inputs = []
        return [l for l in inputs if not l.startswith('params')]

    def _strip_unused_params(self, params):
        used_params = []
        for node in self.graph:
            inputs = node['inputs']
            if inputs is None: inputs = []
            used_params.extend(inputs)

        del_keys = []
        for kk in params:
            if kk not in used_params:
                del_keys.append(kk)
        for key in del_keys:
            del params[key]
        return params

    def _find_input_output_nodes(self, node_infos):
        # node_infos = [node for node in node_infos if node.get('op_type') in self.supported_ops]
        all_node_ids = [node['name'] for node in node_infos]
        all_needed_ids = []
        for node in node_infos:
            all_needed_ids += get_input_node(node)

        if not self.input_node_ids:
            for id_ in all_needed_ids:
                if find_node(node_infos, id_)['inputs'] is None:
                    self.input_node_ids.append(id_)
        assert self.input_node_ids
        if not self.output_node_ids:
            for id_ in all_node_ids:
                if id_ not in all_needed_ids:
                    self.output_node_ids.append(id_)

        for node_name in self.input_node_ids:
            node = find_node(node_infos, node_name)
            if node.get('op_type') == 'Const':
                self.input_node_shapes.append(node['attrs']['shape'])
            else:
                self.input_node_shapes.append(node['attrs']['O_shape'][0])

        unused_node_ids = find_unused_node(node_infos, self.input_node_ids, self.output_node_ids, self.strip_inputs)
        node_infos = [node for node in node_infos if node['name'] not in unused_node_ids]
        return node_infos

