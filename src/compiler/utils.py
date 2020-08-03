import os
import glog as log
import math
from copy import deepcopy

def remove_nodes(node_list, op_type):
    nodes_to_remove = []
    new_node_list = []
    _node_map = {}

    for node in node_list:
        _node_map[node.name] = node
        if node.op_type == op_type:
            nodes_to_remove.append(node)

    identity_to_preserve = set()
    for rm_node in nodes_to_remove:
        rm_node_name = rm_node.name
        rm_node_inputs_id = [_id for _id in rm_node.get_inputs()]
        for _node_idx, node in enumerate(node_list):
            if node.get_inputs() is None:
                continue
            new_inputs_list = []
            for idx, input_name in enumerate(node.get_inputs()):
                if input_name != rm_node_name:
                    new_inputs_list.append(input_name)
                else:
                    if 'index' in rm_node.attrs and _node_map[rm_node.inputs[0]].op_type == 'nn.batch_norm':
                        new_inputs_list.extend(rm_node_inputs_id)
                    else:
                        new_inputs_list.append(input_name)
                        identity_to_preserve.add(rm_node_name)
            node_list[_node_idx].set_inputs(new_inputs_list)
    
    for node in node_list:
        if node.op_type != op_type:
            new_node_list.append(node)
        elif node.name in identity_to_preserve:
            node.attrs['A_shape'] = [_node_map[node.inputs[0]].attrs['O_shape'][node.attrs['index']]]
            node.attrs['O_shape'] = node.attrs['A_shape']
            new_node_list.append(node)

    return new_node_list


def write_tensor(fname, tsr):

    with open(fname,'w') as fpw:

        shp = tsr.shape
        for d in shp:
            fpw.write('{} '.format(d))
        fpw.write('\n')

        tsr_flat = tsr.reshape([-1])
        for i in range(tsr_flat.size):
            if i<tsr_flat.size-1:
                fpw.write( '{} '.format(tsr_flat[i]) )
            else:
                fpw.write( '{}'.format(tsr_flat[i]) )

        fpw.write('\n')


def cast_params(params, dtype='float32'):
    casted_params = {}
    for k, v in params.items():
        casted_params[k] = v.astype(dtype)
    return casted_params


def dump_params(params, params_folder_name, output_path):
    params_folder_path = os.path.join(output_path, params_folder_name)
    if not os.path.isdir(params_folder_path):
        os.mkdir(params_folder_path)
    for k, v in params.items():
        write_tensor(os.path.join(params_folder_path, 'params/'+k[7:].replace('/', '-')), v)


def merge_attrs(attrs_a, attrs_b):
    '''Given two attrs dict merge them into one single dict'''
    if (attrs_a is None) and (attrs_b is not None):
        return attrs_b
    elif (attrs_a is not None) and (attrs_b is None):
        return attrs_a
    elif (attrs_a is None) and (attrs_b is None): 
        log.warning('Attrs can not be None type.')
        return None

    attrs = attrs_a.copy()
    attrs.update(attrs_b)
    return attrs
