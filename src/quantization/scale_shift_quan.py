import tensorflow as tf
import numpy as np
import glog
import os
from ..utils.tf_functions import *
from ..utils import *
from .base_quan import BaseQuan

class ScaleShiftQuan(BaseQuan):
    def __init__(self, weight_quan, image_path, graph, params, table=None, **kwargs):
        super(ScaleShiftQuan, self).__init__(weight_quan, image_path, graph, params, table)

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
                    weight_max = np.absolute(weight_npx).max()
                    shift = get_bit_shift(weight_max)
                elif self.weight_quan == 'perchannel':
                    weight_max = np.absolute(weight_npx).reshape(-1, weight_npx.shape[3]).max(axis=0)
                    shift = get_bit_shift_vec(weight_max)
                else:
                    raise NotImplementedError

                new_weight = quantize_and_dequantize(weight, shift)
                new_weight_npx =  sess.run(new_weight)
                weight_dist = distance(new_weight_npx, weight_npx)
                info.update(dict(weight_name=weight.name, weight_shift=shift, weight_dist=weight_dist))
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
            act_shift = get_bit_shift(max(-act_min, act_max, 1e-5))

            quan_x = quantize(x, act_shift)
            dequan_x = dequantize(quan_x, act_shift)

            dequan_npx = sess.run(dequan_x, feed_dict=feed_dict)
            dist = distance(npx, dequan_npx)
            self.params[despec(dequan_x.name)] = dequan_npx
            info.update(dict(act_shift=act_shift, shape=npx.shape, distance=dist))
            self.info_list.append(info)
            self.act_list.append(dict(node=name, act=dequan_npx))
            glog.info(str(info))
            sess.close()
            return dequan_x
        return wrapper
