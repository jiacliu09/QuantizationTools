import tensorflow as tf
import numpy as np
import glog
import os
from ..utils.tf_functions import *
from ..utils import *
from .base_quan import BaseQuan

spec = lambda x, i : x + ':' + str(i)
class NullQuan(BaseQuan):

    def __init__(self, weight_quan, image_path, graph, params, **kwargs):
        super(NullQuan, self).__init__(weight_quan, image_path, graph, params)

    def _get_calib_table(self, table):
        return None

    def inference(self, function_name):
        function = function_factory[function_name]
        def wrapper(inputs, name, **kwargs):
            sess = tf.Session()
            info = dict(name=name)

            inputs = [self.node_dict[spec(in_, 0)] for in_ in inputs]
            # import pdb; pdb.set_trace()

            feed_dict = self._prepare_feed_dict(function, inputs, name, **kwargs)
            x = function(inputs, name, **kwargs)
            # sess.run
            npx = sess.run(x, feed_dict=feed_dict)
            act_min = npx.min()
            act_max = npx.max()

            info.update(dict(act_min=act_min, act_max=act_max, shape=npx.shape))
            self.params[despec(x.name)] = npx
            self.info_list.append(info)
            self.act_list.append(dict(node=name, act=npx))
            # glog.info(info)
            sess.close()
            return x
        return wrapper
