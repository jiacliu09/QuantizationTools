import numpy as np
import glog as log
import os
from copy import deepcopy
import json

from .op_fuser import OpFuser
from .visualize import visualize_mir_as_graphviz

class Compiler(OpFuser):
    def __init__(self, graph, params=None):
        super(Compiler, self).__init__(graph, params)

    def fuseOperations(self):
        self.merge_Conv_Bias()
        self.fuse_batch_norm()
        self.fuse_bn_to_ScaleBias()

    def mergeOperations(self):
        self.merge_Conv_Bias()
        self.merge_Prev_Relu()
        self.merge_Prev_Clip()
        self.merge_Upsampling()

    def CModel_transforms(self):
        self.markDelayStride()
        self.defuseSoftmax()

    def BertOperations(self):
        self.fuse_constants_transpose()
        self.merge_Gelu()

    def compile(self):
        log.info('Starting compile.')
        self.fuseOperations()
        self.mergeOperations()
        log.info('Compile success!')

    def save(self, graph_name, params_name, output_path):
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        graph_file_name = os.path.join(output_path, graph_name) + '.json'
        params_file_name = os.path.join(output_path, params_name)
        visualize_mir_as_graphviz(self.graph, graph_name, output_path)
        self.save_graph(graph_file_name)
        self.save_params(params_file_name)
        log.info('Save success! Graph {} and Params {} saved at {}.'.format(graph_name, params_name, output_path))
