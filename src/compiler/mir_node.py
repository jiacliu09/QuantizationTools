import numpy as np
import glog as log
from copy import deepcopy

from .op_type import Supported_Op_Type, Mir_Op_Type, Might_Support_Op_Type

class Mir_Node(object):
    def __init__(self):
        self.name = None  # str id
        self.op_type = None
        self.inputs = None
        self.attrs = {}

    def set_node_from_relay(self, node_id, relayviz_node, graph_inputs=None):
        self.name = str(node_id)
        if relayviz_node['node_kind'] == 'Var':
            self.set_op_type('Const')
            self.set_param_name(relayviz_node['name'], graph_inputs)
            self.attrs['shape'] = tuple(relayviz_node['shape'])
            self.attrs['dtype'] = relayviz_node['dtype']
        elif relayviz_node['node_kind'] == 'Call':
            self.set_op_type(relayviz_node['op'])
            self.inputs =  [str(arg) for arg in relayviz_node['args']]
            self.attrs = relayviz_node['attrs']
            self.attrs['A_shape'] = relayviz_node['type_args']
            self.attrs['O_shape'] = relayviz_node['checked_type']
        elif relayviz_node['node_kind'] == 'TupleGetItem':
            self.set_op_type('Identity')
            self.inputs = [str(relayviz_node['tuple_value'])]
            self.attrs['index'] = relayviz_node['index']
        elif relayviz_node['node_kind'] == 'Tuple':
            self.set_op_type('Identity')
            self.inputs =  [str(arg) for arg in relayviz_node['fields']]
        elif relayviz_node['node_kind'] == 'Constant':
            self.set_op_type('Const')
            self.set_param_name('const_' + self.name)
            self.attrs['shape'] = tuple(relayviz_node['data'].shape)

    def set_op_type(self, op_type):
        if op_type in Supported_Op_Type._value2member_map_:
            self.op_type = op_type
        elif op_type in Might_Support_Op_Type._value2member_map_:
            self.op_type = op_type
            log.warn('Might support op type {}'.format(op_type))
        elif op_type in Mir_Op_Type._value2member_map_:
            self.op_type = op_type
        else:
            self.op_type = op_type
            log.warn('Not support op type {}'.format(op_type))

    def set_param_name(self, param_name, graph_inputs=None):
        if graph_inputs and param_name in graph_inputs:
            self.param_name = 'input/' + str(param_name)
            return
        if self.op_type == 'Const':
            if isinstance(param_name, str):
                self.param_name = 'params/' + param_name
            else:
                self.param_name = 'params/' + str(param_name)
        else:
            log.warn('param name only for params node, not node {}, {}'.format(self.name, self.op_type))

    def set_inputs(self, inputs):
        if isinstance(inputs, list):
            self.inputs = inputs
        else:
            log.error('inputs of node {} is not a list.'.format(self.name))

    def get_inputs(self):
        if self.inputs is None:
            return None
        return deepcopy(self.inputs)

    def set_attrs(self, item):
        if isinstance(item, dict):
            self.attrs = item
        else:
            log.error('attrs of node {} is not a dict.'.format(self.name))

    def get_attrs(self, key=None):
        if key is None:
            return deepcopy(self.attrs)
        else:
            if key not in self.attrs:
                return None
            return deepcopy(self.attrs[key])

