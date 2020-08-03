import os
import numpy as np
import glog as log
import json

from .utils import merge_attrs
from .transformer import Transformer
from .mir_node import Mir_Node


class OpFuser(Transformer):
    def __init__(self, graph, params=None):
        super(OpFuser, self).__init__(graph , params)
        self._fold_constants()

    def fuse_batch_norm(self):
        pattern = ['nn.batch_norm', [['nn.conv2d|mir.conv2d_bias']]]
        matches = self.GetOptypeMatches(pattern)

        if matches is None:
            return

        generate_nodes = []
        for _m in matches:
            bn_node = _m[0]
            conv_node = _m[1]

            bn_node_inputs = bn_node.get_inputs()
            gamma_values = self.params[bn_node_inputs[1]]
            beta_values = self.params[bn_node_inputs[2]]
            mean_values = self.params[bn_node_inputs[3]]
            var_values = self.params[bn_node_inputs[4]]
            epsilon = bn_node.get_attrs('epsilon')
            self.remove_params(bn_node_inputs[1:]) # Remove beta gamma mean var in params.

            conv_node_inputs = conv_node.get_inputs()
            conv_weights = self.params[conv_node_inputs[1]]
            if conv_node.op_type == 'mir.conv2d_bias':
                conv_bias = self.params[conv_node_inputs[2]]

            scale_values = (1.0 / np.sqrt(var_values + epsilon)) * gamma_values
            offset_values = (-mean_values * scale_values) + beta_values

            weights_shape = conv_weights.shape
            kernel_layout = conv_node.get_attrs("kernel_layout")
            if kernel_layout == "HWIO": # tensorflow NHWC
                out_channel = weights_shape[3]
                scaled_weights = scale_values.reshape(1, out_channel) * np.reshape(conv_weights, (-1, out_channel))
                scaled_weights = np.reshape(scaled_weights, weights_shape)
            elif kernel_layout == "OIHW":
                out_channel = weights_shape[0]
                scaled_weights = scale_values.reshape(out_channel, 1) * np.reshape(conv_weights, (out_channel, -1))
                scaled_weights = np.reshape(scaled_weights, weights_shape)
            else:
                raise RuntimeError('No such layout.')

            if conv_node.op_type == 'nn.conv2d':
                bias_value = offset_values
            elif conv_node.op_type == 'mir.conv2d_bias':
                bias_value = scale_values * conv_bias + offset_values

            bias_node = Mir_Node()
            bias_node.name = bn_node.name
            bias_node.set_op_type('nn.bias_add')
            bias_node_param_name = bn_node_inputs[1] + '_bias'
            bias_node_inputs = [bn_node_inputs[0], bias_node_param_name]
            bias_node.set_inputs(bias_node_inputs)
            bias_node.set_attrs(bn_node.get_attrs())
            bias_node.attrs['A_shape'] = bn_node.get_attrs('A_shape')[0:2]

            self.set_param(conv_node_inputs[1], scaled_weights)
            self.set_param(bias_node_param_name, bias_value)
            generate_nodes.append(tuple([bias_node, conv_node]))

        self.replaceMatchingOpTypes(matches, generate_nodes)

    def fuse_bn_to_ScaleBias(self):
        pattern = ['nn.batch_norm']
        matches = self.GetOptypeMatches(pattern)

        if matches is None:
            return

        generate_nodes = []
        for _m in matches:
            bn_node = _m[0]

            bn_node_inputs = bn_node.get_inputs()
            gamma_values = self.params[bn_node_inputs[1]]
            beta_values = self.params[bn_node_inputs[2]]
            mean_values = self.params[bn_node_inputs[3]]
            var_values  = self.params[bn_node_inputs[4]]
            epsilon = bn_node.get_attrs('epsilon')
            self.remove_params(bn_node_inputs[1:]) # Remove beta gamma mean var in params.
            gamma_param_name = bn_node_inputs[1]

            scale_values = (1.0 / np.sqrt(var_values + epsilon)) * gamma_values
            offset_values = (-mean_values * scale_values) + beta_values

            node = Mir_Node()
            node.name = bn_node.name
            node.set_op_type('mir.scale_bias')
            node_inputs = [bn_node_inputs[0], gamma_param_name+'_scale', gamma_param_name+'_bias_add'] # input, scale, bias
            node_ashape = [bn_node.attrs['A_shape'][0], bn_node.attrs['A_shape'][1], bn_node.attrs['A_shape'][3]]
            node.set_inputs(node_inputs)
            node.set_attrs(bn_node.get_attrs())
            node.attrs['A_shape'] = node_ashape

            self.set_param(gamma_param_name+'_scale', scale_values)  # Add scale values to params dict.
            self.set_param(gamma_param_name+'_bias_add', offset_values)  # Add offset values to param dict.
            generate_nodes.append(tuple([node]))

        self.replaceMatchingOpTypes(matches, generate_nodes)

    def merge_Conv_Bias(self):
        ''' Generate Moffett IR for compiler.
            Fuse [nn.bias_add, nn.conv2d] into one operation. Params do not change.
        '''
        pattern = ['nn.bias_add', [['nn.conv2d|mir.conv2d_bias']]]
        matches = self.GetOptypeMatches(pattern)

        if matches is None:
            return

        generate_nodes = []
        for match_pair in matches:
            bias_node = match_pair[0]
            conv_node = match_pair[1]

            node = Mir_Node()
            node.name = bias_node.name
            node.set_op_type('mir.conv2d_bias')
            if conv_node.op_type == 'nn.conv2d':
                node_inputs = conv_node.get_inputs() + bias_node.get_inputs()[1:]
                node_ashape = conv_node.attrs['A_shape'] + bias_node.attrs['A_shape'][1:]
            elif conv_node.op_type == 'mir.conv2d_bias':
                node_inputs = conv_node.get_inputs()[:2] + bias_node.get_inputs()[1:]
                node_ashape = conv_node.attrs['A_shape'][:2] + bias_node.attrs['A_shape'][1:]
            node.set_inputs(node_inputs)
            axis = bias_node.get_attrs('axis')
            node.set_attrs(merge_attrs(conv_node.get_attrs(), {'axis': axis}))
            node.attrs['A_shape'] = node_ashape

            generate_nodes.append(tuple([node]))

        self.replaceMatchingOpTypes(matches, generate_nodes)

    def merge_ConvBias_Add(self):
        ''' Generate Moffett IR for compiler.
            Fuse [add, nn.conv2d|mir.conv2d_bias|mir.scale_bias] into one operation.
            The inputs of node is [prev_feature_map, prev_residual_node, weights_name, bias_name].
        '''
        pattern = ['add', [['nn.conv2d|mir.conv2d_bias|mir.scale_bias'], ['*']]]
        matches = self.GetOptypeMatches(pattern)

        if matches is None:
            return

        while(matches):
            generate_nodes = []
            for match_pair in matches:
                add_node = match_pair[0]
                convbias_node = match_pair[1]
                residual_node = match_pair[2]

                op_type_name = convbias_node.get_op_type().split('.')[-1]
                op_type_name = 'mir.' + op_type_name + '_add'

                node = Mir_Node()
                node.name = add_node.name
                node.set_op_type(op_type_name)
                node_inputs = [convbias_node.get_inputs()[0], residual_node.name] + convbias_node.get_inputs()[1:]
                node_ashape = [convbias_node.get_attrs('A_shape')[0], residual_node.get_attrs('O_shape')[0]] + convbias_node.get_attrs('A_shape')[1:]
                node.set_inputs(node_inputs)
                node.set_attrs(merge_attrs(convbias_node.get_attrs(), add_node.get_attrs()))
                node.attrs['A_shape'] = node_ashape

                generate_nodes.append(tuple([node, residual_node]))

            self.replaceMatchingOpTypes(matches, generate_nodes)
            matches = self.GetOptypeMatches(pattern)

    def merge_Upsampling(self):
        ''' Merge Upsampling operation into previous op as an attributes(upsample, up_scale).
        '''
        pattern = ['nn.upsampling', [['*']]]
        matches = self.GetOptypeMatches(pattern)

        if matches is None:
            return

        generate_node = []
        for match_pair in matches:
            upsample_node = match_pair[0]
            any_node = match_pair[1]

            if any_node.op_type.split('.')[-1] == 'bias_add':
                middle = 'bias'
            else:
                middle = any_node.op_type.split('.')[-1]
            op_type_name = 'mir.' + middle + '_upsampling'
            node = Mir_Node()
            node.name = upsample_node.name
            node.set_op_type(op_type_name)
            node.set_inputs(any_node.get_inputs())
            input_shape = any_node.get_attrs('A_shape')
            node.set_attrs(merge_attrs(any_node.get_attrs(), upsample_node.get_attrs()))
            node.attrs['A_shape'] = input_shape

            generate_node.append(tuple([node]))

        self.replaceMatchingOpTypes(matches, generate_node)

    def merge_Prev_Relu(self):
        ''' Generate Moffett IR for compiler.
            Fuse [nn.relu, conv2d_Add|ConvBias_Add|ScaleAdd_Add] into one operation.
            The inputs of node is [prev_feature_map].
        '''
        pattern = ['nn.relu', [['*']]]
        matches = self.GetOptypeMatches(pattern)

        matches = self.checkMatchesPrerequest(matches)
        if matches is None:
            return

        generate_nodes = []
        for match_pair in matches:
            relu_node = match_pair[0]
            prev_node = match_pair[1]

            if prev_node.op_type.split('.')[-1] == 'bias_add':
                middle = 'bias'
            else:
                middle = prev_node.op_type.split('.')[-1]
            op_type_name = 'mir.' + middle + '_relu'
            node = Mir_Node()
            node.name = relu_node.name
            node.set_op_type(op_type_name)
            node.set_inputs(prev_node.get_inputs())
            node_ashape = prev_node.get_attrs('A_shape')
            node.set_attrs(merge_attrs(prev_node.get_attrs(), relu_node.get_attrs()))
            node.attrs['A_shape'] = node_ashape

            generate_nodes.append(tuple([node]))

        self.replaceMatchingOpTypes(matches, generate_nodes)

    def merge_Prev_Clip(self):
        ''' Generate Moffett IR for compiler.
            Fuse [nn.clip, conv2d_Add|ConvBias_Add|ScaleAdd_Add] into one operation.
            The inputs of node is [prev_feature_map].
        '''
        pattern = ['clip', [['*']]]
        matches = self.GetOptypeMatches(pattern)

        if matches is None:
            return

        generate_nodes = []
        for match_pair in matches:
            clip_node = match_pair[0]
            prev_node = match_pair[1]

            if prev_node.op_type.split('.')[-1] == 'bias_add':
                middle = 'bias'
            else:
                middle = prev_node.op_type.split('.')[-1]
            op_type_name = 'mir.' + middle + '_clip'
            node = Mir_Node()
            node.name = clip_node.name
            node.set_op_type(op_type_name)
            node.set_inputs(prev_node.get_inputs())
            node_ashape = prev_node.get_attrs('A_shape')
            node.set_attrs(merge_attrs(prev_node.get_attrs(), clip_node.get_attrs()))
            node.attrs['A_shape'] = node_ashape

            generate_nodes.append(tuple([node]))

        self.replaceMatchingOpTypes(matches, generate_nodes)

    def markDelayStride(self):
        strides_flag = False
        for idx, node in enumerate(self.graph):
            if not strides_flag and node.op_type == 'nn.max_pool2d':
                strides_flag = True
            if 'conv2d' in node.op_type:
                if node.attrs['strides'][0] == 2 and strides_flag:
                    self.graph[idx].attrs['strides'] = [1, 2]
                    o_shape = node.get_attrs('O_shape')[0]
                    layout = node.attrs['data_layout']
                    if layout == 'NHWC':
                        o_shape = [o_shape[0], o_shape[1]*2, o_shape[2], o_shape[3]]
                    elif layout == 'NCHW':
                        o_shape = [o_shape[0], o_shape[1], o_shape[2]*2, o_shape[3]]
                    else:
                        raise RuntimeError("No such data layout", layout)
                    self.graph[idx].attrs['O_shape'][0] = o_shape

        for idx, node in enumerate(self.graph):
            if 'add' in node.op_type and node.op_type != 'nn.bias_add':
                left_node = self._node_map[node.inputs[0]]
                right_node = self._node_map[node.inputs[1]]

                delay_stride = False
                left_stride = left_node.attrs['strides'][1] if left_node.attrs and 'strides' in left_node.attrs else 1
                right_stride = right_node.attrs['strides'][1] if right_node.attrs and 'strides' in right_node.attrs else 1

                which_one = None
                if left_stride == 2 or right_stride == 2:
                    delay_stride = True
                    which_one = right_node if left_stride == 2 else left_node

                    for w_idx, w_node in enumerate(self.graph):
                        if which_one.name == w_node.name:
                            o_shape = w_node.get_attrs('O_shape')[0]
                            layout = w_node.attrs['data_layout']
                            if layout == 'NHWC':
                                o_shape = [o_shape[0], o_shape[1]*2, o_shape[2], o_shape[3]]
                            elif layout == 'NCHW':
                                o_shape = [o_shape[0], o_shape[1], o_shape[2]*2, o_shape[3]]
                            else:
                                raise RuntimeError("No such data layout", layout)
                            self.graph[w_idx].attrs['A_shape'][0] = o_shape
                            self.graph[w_idx].attrs['O_shape'][0] = o_shape

                layout = left_node.attrs['data_layout'] if left_node.attrs and 'data_layout' in left_node.attrs else None
                if not layout:
                    layout = right_node.attrs['data_layout'] if right_node.attrs and 'data_layout' in right_node.attrs else None
                assert layout in ['NHWC', 'NCHW']

                if 'conv2d' in left_node.op_type and 'conv2d' in right_node.op_type and delay_stride:
                    self.graph[idx].attrs['strides'] = [2, 1]
                    self.graph[idx].attrs['data_layout'] = layout
                    a_shape = node.get_attrs('A_shape')[0]
                    if layout == 'NHWC':
                        a_shape = [a_shape[0], a_shape[1]*2, a_shape[2], a_shape[3]]
                    elif layout == 'NCHW':
                        a_shape = [a_shape[0], a_shape[1], a_shape[2]*2, a_shape[3]]
                    self.graph[idx].attrs['A_shape'][0] = a_shape

    def defuseSoftmax(self):
        pattern = ['nn.softmax']
        matches = self.GetOptypeMatches(pattern)

        if matches is None:
            return

        generate_nodes = []
        for match_pair in matches:
            softmax_node = match_pair[0]

            exp_node = Mir_Node()
            exp_node.name = softmax_node.name + '_exp'
            exp_node.set_op_type('exp')
            exp_node.set_inputs(softmax_node.get_inputs())
            exp_oshape = softmax_node.attrs['A_shape']
            axis = softmax_node.attrs.pop('axis', None)
            exp_node.set_attrs(softmax_node.get_attrs())
            exp_node.attrs['O_shape'] = exp_oshape

            sum_node = Mir_Node()
            sum_node.name = softmax_node.name + '_sum'
            sum_node.set_op_type('sum')
            sum_node.set_inputs([exp_node.name])
            sum_node.attrs['axis'] = axis
            sum_node.attrs['A_shape'] = exp_node.attrs['O_shape']
            sum_oshape = list(exp_node.get_attrs('O_shape')[0])
            del sum_oshape[axis]
            sum_node.attrs['O_shape'] = [tuple(sum_oshape)]

            reciprocal_node = Mir_Node()
            reciprocal_node.name = softmax_node.name + '_reciprocal'
            reciprocal_node.set_op_type('reciprocal')
            reciprocal_node.set_inputs([sum_node.name])
            reciprocal_node.attrs['A_shape'] = sum_node.attrs['O_shape']
            reciprocal_node.attrs['O_shape'] = reciprocal_node.attrs['A_shape']

            multiply_node = Mir_Node()
            multiply_node.name = softmax_node.name
            multiply_node.set_op_type('multiply')
            multiply_node.set_inputs([exp_node.name, reciprocal_node.name])
            multiply_node.attrs['A_shape'] = [exp_node.attrs['O_shape'][0], reciprocal_node.attrs['O_shape'][0]]
            multiply_node.attrs['O_shape'] = softmax_node.attrs['O_shape']

            generate_nodes.append(tuple([multiply_node, reciprocal_node, sum_node, exp_node]))
        self.replaceMatchingOpTypes(matches, generate_nodes)

    def fuseSoftmax(self):
        pattern = ['multiply', [['exp'], ['reciprocal', [['sum', [['exp']]]]]]]
        matches = self.GetOptypeMatches(pattern)

        if matches is None:
            return

        generate_nodes = []
        for match_pair in matches:
            last_node = match_pair[0]
            first_node = match_pair[-1]
            sum_node = match_pair[-2]

            softmax_node = Mir_Node()
            softmax_node.name = last_node.name
            softmax_node.set_op_type('nn.softmax')
            softmax_node.set_inputs(first_node.get_inputs())
            softmax_node.attrs['axis'] = sum_node.get_attrs('axis')
            softmax_node.attrs['A_shape'] = first_node.get_attrs('A_shape')
            softmax_node.attrs['O_shape'] = last_node.get_attrs('O_shape')

            generate_nodes.append(tuple([softmax_node]))
        self.replaceMatchingOpTypes(matches, generate_nodes)

    def merge_Gelu(self):
        pattern = ['multiply', [['add'], ['multiply', [['add', [['tanh', [['multiply',[['add', [['add'], ['multiply', [['power', [['add']]]]]]]]]]]]]]]]]
        matches = self.GetOptypeMatches(pattern)

        if matches is None:
            return

        generate_nodes = []
        for match_pair in matches:
            last_node = match_pair[0]
            input_node = match_pair[-1]

            gelu_node = Mir_Node()
            gelu_node.name = last_node.name
            gelu_node.set_op_type('gelu')
            gelu_node.set_inputs([input_node.name])
            gelu_node.attrs['A_shape'] = input_node.attrs['O_shape']
            gelu_node.attrs['O_shape'] = last_node.attrs['O_shape']

            generate_nodes.append(tuple([gelu_node, input_node]))
        self.replaceMatchingOpTypes(matches, generate_nodes)

    def fuse_constants_transpose(self):
        pattern = ['nn.dense', [['transpose'], ['*']]]
        matches = self.GetOptypeMatches(pattern)

        matches = self.checkTransposeMatchesPrerequest(matches)
        if matches is None:
            return

        while(matches):
            generate_nodes = []
            for match_pair in matches:
                dense_node = match_pair[0]
                trans_node = match_pair[1]
                prev_node = match_pair[2]

                trans_node_inputs = trans_node.get_inputs()
                constants = self.params[trans_node_inputs[0]]
                self.remove_params(trans_node_inputs)
                axes = trans_node.attrs['axes']
                dense_weight = np.transpose(constants, axes)

                new_node = Mir_Node()
                new_node.name = dense_node.name
                new_node.set_op_type('nn.dense')
                dense_node_inputs = dense_node.get_inputs()
                dense_node_inputs.remove(trans_node.name)
                new_node.set_inputs([dense_node_inputs[0], trans_node_inputs[0]])
                new_node.set_attrs(dense_node.get_attrs())

                self.set_param(trans_node_inputs[0], dense_weight)
                generate_nodes.append(tuple([new_node, prev_node]))
            self.replaceMatchingOpTypes(matches, generate_nodes)
            matches = self.GetOptypeMatches(pattern)
            matches = self.checkTransposeMatchesPrerequest(matches)

