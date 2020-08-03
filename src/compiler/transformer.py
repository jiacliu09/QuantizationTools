import json
from functools import cmp_to_key
import glog as log
from copy import deepcopy
import os.path as path

class Transformer(object):
    def __init__(self, graph, params=None):
        self.graph = deepcopy(graph)
        self._node_map = self.__init_node_map(self.graph)
        if not params:
            self.params = params
        else:
            self.params = {'params/'+k: v.asnumpy() for k, v in params.items()}

    def __init_node_map(self, graph):
        _node_map = {}
        for node in graph:
            _node_map[node.name] = node
        return _node_map

    def __reset_node_map(self):
        _node_map = {}
        for node in self.graph:
            _node_map[node.name] = node
        self._node_map = _node_map

    def __remove_nodes_from_graph(self, nodes_list):
        for _n in nodes_list:
            self.graph.remove(_n)

    def __remove_params(self, param_names):
        for p in param_names:
            self.__remove_param(p)

    def __remove_param(self, param_name):
        del self.params[param_name]

    def __set_param(self, param_name, param_val):
        self.params[param_name] = param_val

    def remove_params(self, param_names):
        self.__remove_params(param_names)

    def set_param(self, param_name, param_val):
        self.__set_param(param_name, param_val)

    def get_graph(self):
        return self.graph

    def get_model(self):
        return self.graph.copy(), self.params.copy()

    def save_graph(self, graph_path):
        with open(graph_path, 'w') as f:
            json.dump(self.graph, f, indent=2, separators=(',', ': '), default=lambda x: x.__dict__)

    def save_params(self, params_path):
        import numpy as np
        np.savez(params_path, self.params)

    def _fold_constants(self):
        _to_removed_nodes = set()

        for idx in range(len(self.graph)):
            node = self.graph[idx]

            node_inputs = node.get_inputs()
            if node_inputs is not None:
                for i in range(len(node_inputs)):
                    if node_inputs[i] in self._node_map:
                        _aux_node = self._node_map[node_inputs[i]]
                    else:
                        _aux_node = None
                    if _aux_node and _aux_node.op_type == 'Const' and not _aux_node.param_name.startswith('input/'):
                        _aux_param_name = _aux_node.param_name
                        node_inputs[i] = _aux_param_name
                        _to_removed_nodes.add(_aux_node)
                self.graph[idx].inputs = node_inputs
            else:
                continue
            
        self.__remove_nodes_from_graph(_to_removed_nodes)

    def __RecordMatchedNodes(self, match, matched_nodes):
        for match_node in match:
            matched_nodes.add(match_node.name)

    def __DoesOpTypeMatch(self, node, pattern, previously_matched_nodes, match):
        if node.name in previously_matched_nodes:
            return False
        
        pattern_matched = False
        if pattern[0] == '*':
            pattern_matched = True
        else:
            pattern_ops = pattern[0].split('|')
            for pattern_op in pattern_ops:
                if node.op_type == pattern_op:
                    pattern_matched = True
        
        if not pattern_matched:
            return False

        match.append(node)

        if len(pattern[1:]) == 0:
            # If there are no inputs, assume that's the end of the pattern.
            return True
        
        if node.inputs is None:
            log.error('node ( {} ) have no inputs'.format(node.name))
            return False

        non_control_inputs = []
        for input_node_id in node.inputs:
            if not input_node_id.startswith('params/'):
                non_control_inputs.append(input_node_id)

        pattern_input_size = len(pattern[1])
        if len(non_control_inputs) != pattern_input_size:
            log.error('node ({}) inputs size is not equal to the size of inputs given in pattern'.format(node.name))
            return False

        matched_input_set = set()
        for i in range(pattern_input_size):
            input_pattern = pattern[1][i]
            for j in range(len(non_control_inputs)):
                if j in matched_input_set:
                    continue
                input_node = self._node_map[non_control_inputs[j]]
                if not self.__DoesOpTypeMatch(input_node, input_pattern, previously_matched_nodes, match):
                    continue
                else:
                    matched_input_set.add(j)
                    break
            if len(matched_input_set) != i+1:
                return False

        return True

    def GetOptypeMatches(self, pattern):
        '''Search nodes which match the pattern.
           If there is '*' in pattern, please write it as the last input of following node.
           For example, pattern = ['add', [['nn.conv2d|ConvBias|ScaleAdd'], ['*']]]
        '''
        matched_nodes = set()
        matched_pattern_list = []
        for node in self.graph:
            match = []
            if node.name in matched_nodes:
                continue
            if self.__DoesOpTypeMatch(node, pattern, matched_nodes, match):
                self.__RecordMatchedNodes(match, matched_nodes)
                matched_pattern_list.append(match)
        if matched_pattern_list == []:
            return None
        return matched_pattern_list
    
    def replaceMatchingOpTypes(self, matches, transformed_matches):
        matched_id = []
        for match_pair in matches:
            for node in match_pair:
                matched_id.append(node.name)

        assert len(matches) == len(transformed_matches)
        for match_pair, trans_pair in zip(matches, transformed_matches):
            if match_pair[0].name != trans_pair[0].name:
                log.error('Expected {} to be preserved!'.format(match_pair[0]))
                raise RuntimeError('Node expected to be preserved.')
        
        self.graph = remove_matches_from_graph(self.graph, matches)
        self.graph = insert_matches_into_graph(self.graph, transformed_matches)
        self.__reset_node_map()

    def checkMatchesPrerequest(self, matches):
        if matches is None:
            return
        
        not_matches = []
        for match_pair in matches:
            for node in self.graph:
                if node.op_type == 'Const':
                    continue
                elif match_pair[1].name in node.inputs and node.name != match_pair[0].name:
                    not_matches.append(match_pair)
                    break
        real_matches = []
        for match_pair in matches:
            if match_pair not in not_matches:
                real_matches.append(match_pair)

        return real_matches

    def checkTransposeMatchesPrerequest(self, matches):
        if matches is None:
            return

        real_matches = []
        for match_pair in matches:
            if len(match_pair[1].inputs) != 1 or not match_pair[1].inputs[0].startswith('params/'):
                continue
            else:
                real_matches.append(match_pair)
        return real_matches

def remove_nodes_from_graph(graph, nodes):
    for node in nodes:
        graph.remove(node)
    return graph


def remove_matches_from_graph(graph, matches):
    match_nodes = set()
    for match_pair in matches:
        for node in match_pair:
            match_nodes.add(node)
    return remove_nodes_from_graph(graph, match_nodes)


def insert_nodes_into_graph(graph, nodes):
    _graph = [n for n in graph]
    for node in nodes:
        _graph.append(node)
    _graph = SortByExecutionOrder(_graph)  # Sort nodes in graph by execution order
    return _graph


def insert_matches_into_graph(graph, matches):
    match_nodes = []
    for match in matches:
        for node in match:
            match_nodes.append(node)
    return insert_nodes_into_graph(graph, match_nodes)


def SortByExecutionOrder(graph):
    _graph = deepcopy(graph)
    node_dict = dict()
    for n in graph:
        node_dict[n.name] = n

    visited = set()
    ret_list = list()
    used_node_subset = None
    for _, n in node_dict.items():
        if n.op_type == 'Const':
            visited.add(n.name)
            ret_list.append(n)
            continue
        if n.name not in visited:
            _toposort(n, visited, ret_list, node_dict, used_node_subset)
    
    return ret_list


def _toposort(node, visited, ret_list, node_dict, used_node_subset):
    for n_id in node.inputs:
        if not n_id.startswith('params/') and n_id not in visited:
            _toposort(node_dict[n_id], visited, ret_list, node_dict, used_node_subset)
    
    visited.add(node.name)
    if used_node_subset==None or node.name in used_node_subset:
        ret_list.append(node)
