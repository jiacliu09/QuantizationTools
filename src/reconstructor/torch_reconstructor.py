import torch
import torch.nn as nn
import numpy as np
from .reconstructor import BaseReconstructor, find_node
from ..utils.fake_torchmodule import module_factory

class TorchReconstructor(BaseReconstructor, nn.Module):

    def __init__(self, graph, params=None, input_node_ids=[], output_node_ids=[], update_node_cfg=[]):
        super(TorchReconstructor, self).__init__(
                graph, params, input_node_ids, output_node_ids, update_node_cfg)
        self.model = torch.nn.ModuleList()
        self.bias_register = self._registe_bias()
        self._parse_model()

    def _parse_model(self):
        for gnode in self.graph:
            op_type = gnode.get('op_type')
            op_name = gnode.get('name')
            if op_name in self.bias_register:
                biasnode = find_node(self.graph, self.bias_register[op_name])
                self.model.append(module_factory[op_type](gnode, self.params, biasnode))
            else:
                self.model.append(module_factory[op_type](gnode, self.params))

    def load_weights(self):
        if self.params is not None:
            for m in self.model:
                m.load_weights()

    def _registe_bias(self):
        tmp = {}
        for node in self.graph:
            if node.get('op_type') != 'nn.bias_add':
                continue
            input_node_id = node['inputs'][0]
            tmp[input_node_id] = node['name']
        return tmp

    def _registe_const(self, outs):
        for name in self.params:
            if name.startswith('params/const'):
                outs[name] = torch.from_numpy(self.params[name])


    def forward(self, *args):
        outs = dict((name, x) for (name , x) in zip(self.input_node_ids, args))
        self._registe_const(outs)

        for module in self.model:
            tsrs_in = [outs[name] for name in module.real_inputs]
            y = module(*tsrs_in)
            outs[module.name] = y
        y = [outs[name] for name in self.output_node_ids]
        if len(y) > 1:
            return y
        else:
            return y[0]

