import tensorflow as tf
import numpy as np
import glog
import os
from ..utils.tf_functions import *
from ..utils.tf_quantize_functions import *
from ..utils import *
from .base_quan import BaseQuan

spec = lambda x, i : x + ':' + str(i)
despec = lambda x: x.split(':')[0]

class SymmetryMaxQuan(BaseQuan):

    def __init__(self, weight_quan, image_path, graph, params, table=None, **kwargs):
        super(SymmetryMaxQuan, self).__init__(weight_quan, image_path, graph, params, table)

    def inference(self, function_name):
        function = function_factory[function_name]
        def wrapper(inputs, name, **kwargs):
            sess = tf.Session()

            inputs_tsr = [self.node_dict[spec(in_,0)] for in_ in inputs]
            info = dict(name=name+'_'+function.__name__)
            feed_dict = self._prepare_feed_dict(function, inputs_tsr, name, **kwargs)
            if function in self.Need_quantized_function:
                weight = inputs_tsr[1]
                if len(inputs_tsr) == 3:
                    calib_param = self.table[spec(inputs[2], 0)]
                    bias_max = calib_param['max']
                    bias_sc = bias_max * 2 / 255.
                    inputs_tsr[2] = symmetry_max_quantize_and_dequantize(inputs_tsr[2], bias_sc)

                weight_npx = self.params[despec(weight.name)]
                if self.weight_quan == 'perlayer':
                    calib_param = self.table[spec(inputs[1], 0)]
                    # weight_max = max(calib_param['max'])
                    weight_max = np.absolute(weight_npx).max()
                    scale = weight_max * 2 / 255.
                elif self.weight_quan == 'perchannel':
                    calib_param = self.table[spec(inputs[1], 0)]
                    weight_max = calib_param['max']
                    scale = np.zeros_like(weight_max)
                    for idx, w in enumerate(weight_max):
                        scale[idx] = np.float32(max(w, 1e-5) * 2 / 255.)
                else:
                    raise NotImplementedError

                new_weight = symmetry_max_quantize_and_dequantize(weight, scale)
                # new_weight = quan_ops.quantize_and_dequantize_weights_perchannel(weight, channel_dim_index=3)
                new_weight_npx = sess.run(new_weight)
                weight_dist = distance(new_weight_npx, weight_npx)
                info.update(dict(weight_name=weight.name, weight_scale=scale,
                    weight_max=weight_max, weight_dist=weight_dist))
                inputs_tsr[1] = new_weight

            x = function(inputs_tsr, name, **kwargs)
            try:
                npx = sess.run(x, feed_dict=feed_dict)
            except Exception as err:
                import pdb; pdb.set_trace()
                print()

            if function.__name__ in ['batch_flatten', 'equal']:
                self.params[despec(x.name)] = npx
                self.info_list.append(info)
                self.act_list.append(dict(node=name, act=npx))
                return x

            # act_max = np.absolute(npx).max()
            # act_max = max(-self.table[name][0], self.table[name][1])
            act_max = self.table[spec(name, 0)]['max']
            act_scale = np.float32(max(act_max, 1e-5) / 127.)
            quan_x = symmetry_max_qunatize(x, act_scale)
            dequan_x = symmetry_max_dequantize(quan_x, act_scale)

            # quan_x = quan_ops.quant(x, min=-act_max, max=act_max)
            # dequan_x = quan_ops.de_quant(quan_x, min=-act_max, max=act_max)

            quan_npx = sess.run(quan_x, feed_dict=feed_dict)
            dequan_npx = sess.run(dequan_x, feed_dict=feed_dict)

            dist = distance(npx, dequan_npx)
            self.params[despec(dequan_x.name)] = dequan_npx
            info.update(dict(act_scale=act_scale, act_max=npx.max(), shape=npx.shape,
                distance=dist))
            self.info_list.append(info)
            self.act_list.append(dict(node=name, act=dequan_npx))
            glog.info(str(info))
            sess.close()
            return dequan_x
        return wrapper

class TrueSymmetryMaxQuan(BaseQuan):
    def __init__(self, weight_quan, image_path, graph, params, table=None, **kwargs):
        super(TrueSymmetryMaxQuan, self).__init__(weight_quan, image_path, graph, params, table)

    def inference(self, function_name):
        function = quantize_function_factory[function_name]

        def wrapper(inputs, name, **kwargs):
            sess = tf.Session()
            output_info = []
            for output_name in self.table:
                if despec(output_name) == name:
                    output_info.append(self.table[output_name])
            input_info = []
            for input_name in inputs:
                input_info.append(self.table[spec(input_name, 0)])
            act_max = output_info[0]['max']
            inputs_tsr = [self.node_dict[spec(in_,0)] for in_ in inputs]
            feed_dict = self._prepare_feed_dict(function, inputs_tsr, name, **kwargs)
            rst = function(inputs_tsr, name, input_info=input_info, output_info=output_info, mode=self.weight_quan, **kwargs)
            if kwargs['quantize']:
                rst_npy = sess.run(quan_ops.de_quant(rst, min=-act_max, max=act_max), feed_dict=feed_dict)
            else:
                rst_npy = sess.run(rst, feed_dict=feed_dict)
            self.act_list.append(dict(node=name, act=rst_npy))
            # orig_npy = np.load('temp/null/%s.npy' % name)
            # np.save('temp/true/%s.npy' % name, rst_npy)
            # glog.info("name: {}, distance: {}".format(name, distance(rst_npy, orig_npy)))
            self.params[despec(rst.name)] = sess.run(rst, feed_dict=feed_dict)
            return rst
        return wrapper

