import json
import copy
import sys
import glog as log
from math import ceil, floor


def tile_op(op_node, slice_config, tile_type, is_start_op=False):
    t_op = copy.deepcopy(op_node)

    init_h, init_w = slice_config['init_shape'] # shape of start slice node.
    output_layout = slice_config['output_layout']
    block_shape = slice_config['block_shape']

    if 'slice' == tile_type:
        An, Ah, Aw, Ac = t_op['attr']['O_shape'] # It may palceholder
    else:
        An, Ah, Aw, Ac = t_op['attr']['A_shape']
    On, Oh, Ow, Oc = t_op['attr']['O_shape'] # original output shape

    h_ii_ratio, w_ii_ratio = Ah/init_h, Aw/init_w
    h_oi_ratio, w_oi_ratio = Oh/init_h, Ow/init_w

    need_hw_slice = True
    if Ah <= block_shape[0] and Aw <= block_shape[1]:
        need_hw_slice = False

    block_input_shape = [An,
                         int(block_shape[0] * h_ii_ratio) if need_hw_slice else Ah,
                         int(block_shape[1] * w_ii_ratio) if need_hw_slice else Aw,
                         Ac]

    block_output_shape = [On,
                         int(block_shape[0] * h_oi_ratio) if need_hw_slice else Oh,
                         int(block_shape[1] * w_oi_ratio) if need_hw_slice else Ow,
                         Oc]

    B_layout, H_layout, W_layout = output_layout
    B_grids, H_grids, W_grids = [len(layout) for layout in output_layout]
    total_grids = B_grids * H_grids * W_grids

    gen_ops = list()

    if 'slice' == tile_type and is_start_op:
        gen_ops.append(copy.deepcopy(op_node))

    for batch in range(B_grids):
        for coor_h in range(H_grids):
            for coor_w in range(W_grids):
                t_op = copy.deepcopy(op_node)
                cur_grid = batch * (H_grids * W_grids) + coor_h * W_grids + coor_w
                t_op['name'] = op_node['name'] + "#{}of{}".format(cur_grid, total_grids)
                t_op['attr']['O_shape'] = None

                if 'slice' == tile_type:
                    t_op = dict()
                    t_op['type'] = 'Slice'
                    t_op['name'] = op_node['name'] + "#{}of{}".format(cur_grid, total_grids)
                    t_op['inputs'] = [op_node['name']]
                    t_op['attr'] = dict()
                    t_op['attr']['O_shape'] = [B_layout[batch]] + block_output_shape[1:4]
                    t_op['attr']['slice_from'] = op_node['name']
                    t_op['attr']['slice_from_shape'] = [An, Ah, Aw, Ac]
                    compensation = get_patch_compensation([coor_h, coor_w], block_input_shape[1:3], output_layout[1:3]) # Top, Bottom, Left, Right
                    t_op['attr']['slice_offset'] = [sum(B_layout[:batch]),
                                                    sum(H_layout[:coor_h]) - compensation[0],
                                                    sum(W_layout[:coor_w]) - compensation[2]]
                    log.info('Slice: coor[{} {}]; compensation {}'.format(coor_h, coor_w, compensation))

                    log.info('Slice: coor[{} {}]; compensation {}'.format(coor_h, coor_w, compensation))

                elif 'body' == tile_type:
                    # t_op['name'] = op_node['name'] + "#{}of{}".format(cur_grid, total_grids)
                    t_op['inputs'] = [inp+'#{}of{}'.format(cur_grid, total_grids) if not is_params_node(inp) else inp for inp in op_node['inputs']]
                    t_op['attr']['A_shape'] = [B_layout[batch]] + block_input_shape[1:4]
                    t_op['attr']['O_shape'] = [B_layout[batch]] + block_output_shape[1:4]

                elif 'merge' == tile_type:
                    t_op['inputs'] = [inp+'#{}of{}'.format(cur_grid, total_grids) if not is_params_node(inp) else inp for inp in op_node['inputs']]
                    t_op['attr']['A_shape'] = [B_layout[batch]] + block_input_shape[1:4]
                    t_op['attr']['O_shape'] = [B_layout[batch]] + block_output_shape[1:4]
                    t_op['attr']['merge_to'] = op_node['name']
                    t_op['attr']['merge_to_shape'] = [On, Oh, Ow, Oc]
                    t_op['attr']['merge_offset'] = [sum(B_layout[:batch]),
                                                    int(sum(H_layout[:coor_h]) * h_oi_ratio),
                                                    int(sum(W_layout[:coor_w]) * w_oi_ratio)]
                    new_output_layout = [B_layout,
                                         [int(hl*h_oi_ratio) for hl in H_layout],
                                         [int(wl*w_oi_ratio) for wl in W_layout]]
                    margins = get_patch_margin([coor_h, coor_w], block_output_shape[1:3], new_output_layout[1:3])
                    # t_op['attr']['margins'] = [int(i) for i in margins]
                    margin_attrs = ['top_margin','bottom_margin', 'left_margin', 'right_margin']
                    for _m, _m_attr in zip(margins, margin_attrs):
                        t_op['attr'][_m_attr] = _m
                    log.info('Merge: coor[{} {}]; margins {}'.format(coor_h, coor_w, margins))
                
                gen_ops.append(t_op)

    if 'merge' == tile_type:
        gen_ops.append(insert_merge_op(op_node, total_grids))

    return gen_ops


def is_params_node(node_name):
    return node_name.startswith('params')


def insert_merge_op(op_node, total_grids):
    merge_op = dict()
    merge_op['type'] = 'Merge'
    merge_op['name'] = op_node['name']
    merge_op['inputs'] = [op_node['name'] + '#{}of{}'.format(i, total_grids) for i in range(total_grids)]
    merge_op['attr'] = {'O_shape': op_node['attr']['O_shape']}
    return merge_op


def slice_graph_segment(graph_seg, slice_config, is_start_seg=False):
    slice_op_name = slice_config['slice_from']
    merge_op_name = slice_config['merge_to']
    log.info("Slice from [{}], Merge to [{}]".format(slice_op_name, merge_op_name))


    sliced_graph_seg = list()
    for node in graph_seg:
        if node['name'] == slice_op_name:
            tile_type = 'slice'
        elif node['name'] == merge_op_name:
            tile_type = 'merge'
        else:
            tile_type = 'body'
        sliced_graph_seg.extend(tile_op(node, slice_config, tile_type, is_start_seg))
    return sliced_graph_seg


def slice_graph(graph, slice_config, start_seg_id=0):
    sliced_graph = list()
    seg_id = start_seg_id
    op_id = 0

    while op_id < len(graph):
        slice_from_name = slice_config[seg_id]['slice_from'] if seg_id < len(slice_config) else ""

        if graph[op_id]['name'] == slice_from_name:
            # Take the segment between "slice_from" to "merge_to"
            graph_seg = list()
            while graph[op_id]['name'] != slice_config[seg_id]['merge_to']:
                graph_seg.append(graph[op_id])
                op_id += 1
                assert(op_id < len(graph))

            graph_seg.append(graph[op_id])  # Merge op.
            op_id += 1
            
            cur_slice_config = slice_config[seg_id]
            sliced_subgraph = slice_graph_segment(graph_seg, cur_slice_config, seg_id==0)
            sliced_graph.extend(sliced_subgraph)
            
            if (seg_id < len(slice_config) - 1) and (graph[op_id-1]['name'] == slice_config[seg_id+1]['slice_from']):
                op_id -= 1
            seg_id += 1

        else:
            sliced_graph.append(graph[op_id])
            op_id += 1

    return sliced_graph


def get_patch_compensation(patch_coordinate, block_shape, patch_layout):
    h_layout, w_layout = patch_layout[:]
    h_grids, w_grids = len(h_layout), len(w_layout)
    coor_h, coor_w = patch_coordinate

    is_boundary = [coor_h==0, coor_h==h_grids-1, coor_w==0, coor_w==w_grids-1]  # Top, Bottom, Left, Right
    comp = [int(not is_boundary[0]),
            1 if (is_boundary[0] and is_boundary[1]) else int(not is_boundary[1]),
            int(not is_boundary[2]),
            1 if (is_boundary[2] and is_boundary[3]) else int(not is_boundary[3])]
    # comp = [int(not b) for b in is_boundary]
    
    h_comps = block_shape[0] - h_layout[coor_h]
    w_comps = block_shape[1] - w_layout[coor_w]
    h_div = comp[0] + comp[1]
    w_div = comp[2] + comp[3]
    
    compensation = [0 if h_div == 0 else  ceil(comp[0] * h_comps / h_div), \
                    0 if h_div == 0 else floor(comp[1] * h_comps / h_div), \
                    0 if w_div == 0 else  ceil(comp[2] * w_comps / w_div), \
                    0 if w_div == 0 else floor(comp[3] * w_comps / w_div)]
    return compensation


def get_patch_margin(patch_coordinate, block_shape, patch_layout):
    return get_patch_compensation(patch_coordinate, block_shape, patch_layout)


def main():
    try:
        graph_fpath = sys.argv[1]
        slice_config = sys.argv[2]
        start_seg_id = int(sys.argv[3])
        output_fpath = sys.argv[4]
    except:
        print('slice_graph.py [graph_json] [slice_config] [seg_id] [sliced graph_json]')
        exit(0)

    op_list = json.loads(open(graph_fpath).read())
    slice_config = json.loads(open(slice_config).read())
    
    new_op_list = slice_graph(op_list, slice_config, start_seg_id)
    json.dump(new_op_list, open(output_fpath, 'w'), indent=4, sort_keys=True) 
    print('Done!')


if __name__ == "__main__":
    main()
