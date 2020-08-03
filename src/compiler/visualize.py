from tvm.relay.expr import Expr, Var, Call, TupleGetItem, Tuple, Constant
from tvm.relay.function import Function
from tvm.relay.op.op import Op
from tvm.relay import analysis
from tvm.ir import Attrs
from tvm.relay import op
from tvm.ir.container import Array
from tvm.tir import IntImm
from tvm.runtime.object import Object
from tvm.ir.type import TupleType
from tvm.ir.tensor_type import TensorType
import json

from .utils import remove_nodes
from .mir_node import Mir_Node


def post_order_visit(expr, fvisit):
    """Recursively visit the ir in post DFS order node,
    apply fvisit. Each node is guaranteed to be visited
    only once.
    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression.
    fvisit : function
        The visitor function to be applied.
    """
    return analysis.post_order_visit(expr, fvisit)


def _get_attrs(node_attrs):
    attrs = {}

    def _get_attr_dict(node_attrs, key):
        if isinstance(node_attrs, Attrs):
            att = node_attrs.__getitem__(key)
        elif isinstance(node_attrs, Object):
            att = getattr(node_attrs, key)
        _att = []
        if isinstance(att, Array):
            for i in att:
                if isinstance(i, IntImm):
                    _att.append(int(i.value))
                if isinstance(i, Array):
                    _att.append([int(j.value) for j in i])
            att = _att
        elif isinstance(att, IntImm):
            att = [int(att)]
        return att

    if isinstance(node_attrs, Attrs):
        attrs_keys = node_attrs.keys()
        for k in attrs_keys:
            attrs[k] = _get_attr_dict(node_attrs, k)
    elif isinstance(node_attrs, Object):
        attrs_keys = dir(node_attrs)
        for k in attrs_keys:
            attrs[k] =  _get_attr_dict(node_attrs, k)

    return attrs

def _get_type_args(node_type_args):
    a_shape = []
    for node_type in node_type_args:
        if isinstance(node_type, TensorType):
            a_shape.append(node_type.concrete_shape)
        elif isinstance(node_type, TupleType):
            for field in node_type.fields:
                a_shape.append(field.concrete_shape)
    return a_shape

def _get_checked_type(node):
    o_shape = []
    node_checked_type = node.checked_type
    if isinstance(node_checked_type, TensorType):
        o_shape.append(node_checked_type.concrete_shape)
    elif isinstance(node_checked_type, TupleType):
        if node.op.name == 'nn.batch_norm':
            o_shape.append(node_checked_type.fields[0].concrete_shape)
        else:
            for field in node_checked_type.fields:
                o_shape.append(field.concrete_shape)
    return o_shape

def _export_as_relayviz(expr):
    """Export a Relay function as a nested dictionary, following the RelayViz spec
    (https://discuss.tvm.ai/t/rfc-visualizing-relay-program-as-graph/4825/10). The dictionary will
    contain all information useful for visualizing the Relay program and is meant to be consumed
    by other visualizers.
    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression.
    Returns
    -------
    viz : dict
        Nested dictionary
    """

    # node_dict maps a Relay node to an index (node ID)
    def _traverse_expr(node, node_dict):
        if node in node_dict:
            return
        node_dict[node] = len(node_dict)

    node_dict = {}
    post_order_visit(expr, lambda x: _traverse_expr(x, node_dict))

    relayviz_nodes = []

    # Sort by node ID
    for node, node_idx in sorted(node_dict.items(), key=lambda x: x[1]):
        if isinstance(node, Function):
            relayviz_nodes.append({
                'node_kind': 'Function',
                'body': node_dict[node.body],
                'params': [node_dict[x] for x in node.params]
            })
        elif isinstance(node, Var):
            relayviz_nodes.append({
                'node_kind': 'Var',
                'name': node.name_hint,
                'dtype': node.type_annotation.dtype,
                'shape': [int(x) for x in node.type_annotation.shape]
            })
        elif isinstance(node, Call):
            relayviz_nodes.append({
                'node_kind': 'Call',
                'op': node.op.name,
                'args': [node_dict[arg] for arg in node.args],
                'attrs': _get_attrs(node.attrs) if isinstance(node.attrs, Attrs) or isinstance(node.attrs, Object) else {},
                'type_args': _get_type_args(node.type_args) if isinstance(node.type_args, Array) else [],
                'checked_type': _get_checked_type(node) if isinstance(node.checked_type, TensorType) or isinstance(node.checked_type, TupleType) else []
            })
        elif isinstance(node, Op):
            relayviz_nodes.append({
                'node_kind': 'Op',
                'name': node.name,
            })
        elif isinstance(node, TupleGetItem):
            relayviz_nodes.append({
                'node_kind': 'TupleGetItem',
                'tuple_value': node_dict[node.tuple_value],
                'index': node.index      # The index of output
            })
        elif isinstance(node, Tuple):
            relayviz_nodes.append({
                'node_kind': 'Tuple',
                'fields': [node_dict[field] for field in node.fields],
            })
        elif isinstance(node, Constant):
            relayviz_nodes.append({
                'node_kind': 'Constant',
                'data': node.data,
            })
        else:
            raise RuntimeError(
                'Unknown node type. node_idx: {}, node: {}'.format(node_idx, type(node)))

    obj = {}
    obj['format'] = 'relayviz'
    obj['version'] = [1, 0]
    obj['nodes'] = relayviz_nodes
    return obj


def visualize_mir_as_graphviz(graph, graph_name, output_path):
    from graphviz import Digraph
    dot = Digraph(format='png')
    dot.attr(rankdir='BT')
    dot.attr('node', shape='box')
    for node in graph:
        if node.op_type == 'Const':
            dot.node(node.name,
                     '{} id:{}\n {} shape:{}'.format(node.op_type, node.name, node.param_name[6:],node.attrs['shape']))
        elif node.inputs is not None:
            shape = node.get_attrs('O_shape') if node.get_attrs('O_shape') else '-'
            dot.node(node.name,
                     '{} id:{}\n shape:{}\n'.format(node.op_type, node.name, shape))
            for arg in node.inputs:
                if not arg.startswith('params/'):
                    dot.edge(arg, node.name)
        else:
            raise RuntimeError(
                'Node type {} not supported by GraphViz visualizer.'.format(node.get_op_type()))
    dot.render(filename= graph_name, directory= output_path, cleanup= True)
    return dot


def _export_as_graph_json(relayviz_obj, params, graph_inputs):
    node_list = []

    for node_id, node in enumerate(relayviz_obj['nodes']):
        if node['node_kind'] == 'Var':
            mir_node = Mir_Node()
            mir_node.set_node_from_relay(node_id, node, graph_inputs)
            node_list.append(mir_node)
        elif node['node_kind'] == 'Call':
            mir_node = Mir_Node()
            mir_node.set_node_from_relay(node_id, node)
            node_list.append(mir_node)
        elif node['node_kind'] == 'TupleGetItem':
            mir_node = Mir_Node()
            mir_node.set_node_from_relay(node_id, node)
            node_list.append(mir_node)
        elif node['node_kind'] == 'Tuple':
            mir_node = Mir_Node()
            mir_node.set_node_from_relay(node_id, node)
            node_list.append(mir_node)
        elif node['node_kind'] == 'Constant':
            mir_node = Mir_Node()
            mir_node.set_node_from_relay(node_id, node)
            params['const_'+str(node_id)] = node['data']
            node_list.append(mir_node)
        elif node['node_kind'] == 'Function':
            continue
        elif node['node_kind'] == 'Op':
            pass
        else:
            raise RuntimeError(
                'Node type {} not supported.'.format(node['node_kind']))

    node_list = remove_nodes(node_list, 'Identity')

    return node_list, params


def visualize(model):
    relayviz_obj = _export_as_relayviz(model.mod['main'])
    return _export_as_graph_json(relayviz_obj, model.params, model.graph_inputs)
