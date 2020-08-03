import tensorflow as tf
import numpy as np
import glog
import os
from ..utils.tf_functions import *
from ..utils import *
from .base_quan import BaseQuan
from .quan_ops import quan_ops

# library_path = os.path.join(os.path.dirname(__file__), '_moffett_quantization_ops.so')
# quan_ops = tf.load_op_library(library_path)

class MinMaxQuan(BaseQuan):
    def __init__(self, weight_quan, image_path, graph, params, table=None, **kwargs):
        super(MinMaxQuan, self).__init__(weight_quan, image_path, graph, params, table)

    def inference(self, function_name):
        function = function_factory[function_name]
        def wrapper(inputs, name, **kwargs):
            sess = tf.Session()
            info = dict(name=name)
            feed_dict = self._prepare_feed_dict(function, inputs, name, **kwargs)
            if function in self.Need_quantized_function:
                weight = inputs[1]
                weight_npx = self.params[despec(weight.name)]
                if self.weight_quan == 'perlayer':
                    new_weight = quan_ops.quantize_and_dequantize_weights(weight)
                    weight_max = np.absolute(weight_npx).max()
                elif self.weight_quan == 'perchannel':
                    new_weight = quan_ops.quantize_and_dequantize_weights_perchannel(weight, channel_dim_index=3)
                    weight_max = np.absolute(weight_npx).reshape(-1, weight_npx.shape[3]).max(axis=0)
                else:
                    raise NotImplementedError

                new_weight_npx = sess.run(new_weight)
                weight_dist = distance(new_weight_npx, weight_npx)
                info.update(dict(weight_name=weight.name, weight_max=weight_max, weight_dist=weight_dist))
                inputs[1] = new_weight

            x = function(inputs, name, **kwargs)
            npx = sess.run(x, feed_dict=feed_dict)
            if function == batch_flatten:
                self.params[despec(x.name)] = npx
                self.info_list.append(info)
                self.act_list.append(dict(node=name, act=npx))
                return x

            act_min = self.table[name]['outputs']['min']
            act_max = self.table[name]['outputs']['max']
            quan_x = quan_ops.quant(x, min=act_min, max=act_max)
            dequan_x = quan_ops.de_quant(quan_x, min=act_min, max=act_max)
            dequan_npx = sess.run(dequan_x, feed_dict=feed_dict)

            dist = distance(npx, dequan_npx)
            self.params[despec(dequan_x.name)] = dequan_npx
            info.update(dict(act_min=npx.min(), act_max=npx.max(), shape=npx.shape,
                distance=dist))
            self.info_list.append(info)
            self.act_list.append(dict(node=name, act=dequan_npx))
            glog.info(str(info))
            sess.close()
            return dequan_x
        return wrapper

class TrueMinMaxQuan(BaseQuan):
    def __init__(self, weight_quan, image_path, graph, params, table=None, **kwargs):
        super(TrueMinMaxQuan, self).__init__(weight_quan, image_path, graph, params, table)

    def inference(self, function_name):
        function = function_factory[function_name]
        def wrapper(inputs, name, **kwargs):
            if function in self.Need_quantized_function:
                weight = inputs[1]
                if self.weight_quan == 'perlayer':
                    new_weight = quan_ops.quantize_and_dequantize_weights(weight)
                elif self.weight_quan == 'perchannel':
                    new_weight = quan_ops.quantize_and_dequantize_weights_perchannel(weight, channel_dim_index=3)
                else:
                    raise NotImplementedError
                inputs[1] = new_weight
            if function == batch_flatten:
                x = function(inputs, name, **kwargs)
                return x
            output_min = self.table[name]['outputs']['min']
            output_max = self.table[name]['outputs']['max']
            input_min = self.table[name]['inputs']['min']
            input_max = self.table[name]['inputs']['max']
            # import pdb; pdb.set_trace()
            if function.__name__ == 'sigmoid':
                assert len(inputs) == 1
                rst = quan_ops.quant(inputs[0], min=input_min, max=input_max)
                rst = quan_ops.quantized_activation(
                        rst, input_min=input_min, input_max=input_max,
                        output_min=output_min, output_max=output_max,
                        activation_type=4)
                rst = quan_ops.de_quant(rst, min=output_min, max=output_max)
            elif function.__name__ == 'exp':
                input_min = self.table[name]['inputs']['min']
                input_max = self.table[name]['inputs']['max']
                rst = quan_ops.quant(inputs[0], min=input_min, max=input_max)
                rst = quan_ops.quantized_activation(rst, input_min=input_min, input_max=input_max, output_min=output_min, output_max=output_max, activation_type=5)
                rst = quan_ops.de_quant(rst, min=output_min, max=output_max)
            elif function.__name__ == 'reciprocal':
                input_min = self.table[name]['inputs']['min']
                input_max = self.table[name]['inputs']['max']
                rst = quan_ops.quant(inputs[0], min=input_min, max=input_max)
                rst = quan_ops.quantized_activation(rst, input_min=input_min, input_max=input_max, output_min=output_min, output_max=output_max, activation_type=6)
                rst = quan_ops.de_quant(rst, min=output_min, max=output_max)
            else:
                input_requantize = kwargs.get('input_requantize', False)
                if input_requantize:
                    in0 = quan_ops.quant(inputs[0], min=input_min, max=input_max)
                    in0 = quan_ops.de_quant(in0, min=input_min, max=input_max)
                    inputs[0] = in0
                rst = function(inputs, name, **kwargs)
                requantize = kwargs.get('requantize', False)
                if not requantize:
                    rst = quan_ops.quant(rst, min=output_min, max=output_max)
                    rst = quan_ops.de_quant(rst, min=output_min, max=output_max)
            return rst
        return wrapper
