import tensorflow as tf
import numpy as np
from ..quantization.quan_ops import quan_ops
from ..utils import symmetry_max_qunatize


def const(inputs, name, shape=(1,3,224,224), input_info=None, output_info=None, **kwargs) :
    b, c, h, w = shape
    tx = tf.placeholder(dtype=tf.float32, shape=(None, h, w, c), name=name)
    minv = output_info[0]['min']
    maxv = output_info[0]['max']
    act_max = max(-minv, maxv)
    # tx = quan_ops.quant(tx, min=-act_max, max=act_max)
    return tx

def mir_conv2d_bias_relu(inputs, name, strides=(1,1), padding=(0,0,0,0),
        dilation=(1,1), groups=1, channels=None, input_info=None, output_info=None, mode='perlayer', **kwargs):
    x, weight, bias = inputs


    fmax = input_info[0]['max']
    fscale = fmax * 2 / 255

    if mode == 'perlayer':
        wmax = max(input_info[1]['max'])
        wscale = wmax * 2 / 255
    elif mode == 'perchannel':
        wmax = input_info[1]['max']
        wscale = [max(w, 1e-5) * 2 / 255 for w in wmax]
    else:
        raise

    bmax = input_info[2]['max']
    bscale = bmax * 2 / 255

    act_max = output_info[0]['max']
    act_scale = act_max * 2 / 255
    # import pdb; pdb.set_trace()
    # x = tf.round(x / fscale)
    # weight = tf.round(weight / wscale)
    # bias = tf.round(bias / fscale / wscale)
    if not kwargs['input_requantize'][0]:
        # import pdb; pdb.set_trace()
        x = quan_ops.quant(x, -fmax, fmax)

    # import pdb; pdb.set_trace()
    if mode == 'perlayer':
        weight = quan_ops.quant(weight, -wmax, wmax)
    else:
        neg_wmax = [-val for val in wmax]
        weight = quan_ops.quant_perchannel(weight, neg_wmax, wmax, channel_dim_index=3)

    bias = quan_ops.quant(bias, -bmax, bmax)
    u, l, b, r = padding
    if u + b + l + r > 0:
        x = tf.pad(x, [[0,0], [u,b], [l,r], [0,0]])

    conv_x = tf.nn.conv2d(tf.cast(x, tf.float32),
            tf.cast(weight, tf.float32), strides, padding='VALID', dilations=dilation, name=name)
    # if name == '301':
    #     import pdb; pdb.set_trace()
    if kwargs['use_v2']:
        if mode == 'perlayer':
            # print(name, bscale/(fscale*wscale))
            bias = quan_ops.alignment_to_high_v2(bias,bscale/(fscale*wscale) )
        elif mode == 'perchannel':
            scale = [bscale / (fscale * s) for s in wscale]
            # print(name, scale)
            bias = quan_ops.alignment_to_high_perchannel_v2(bias, scale, data_format='NHWC')
        else:
            raise
    else:
        if mode == 'perlayer':
            bias = quan_ops.alignment_to_high(bias, bscale/(fscale*wscale) )
        elif mode == 'perchannel':
            scale = [bscale / (fscale * s) for s in wscale]
            bias = quan_ops.alignment_to_high_perchannel(bias, scale, data_format='NHWC')
        else:
            raise
    # bias = tf.cast(bias, tf.float32) * bscale / fscale / wscale
    conv_x = tf.nn.bias_add(conv_x, tf.cast(bias, tf.float32))
    conv_x = tf.nn.relu(conv_x)
    if mode == 'perlayer':
        # import pdb; pdb.set_trace()
        if kwargs['use_v2']:
            conv_x = quan_ops.alignment_v2(tf.cast(conv_x, tf.int32), fscale*wscale/act_scale)
        else:
            conv_x = quan_ops.alignment(tf.cast(conv_x, tf.int32), fscale*wscale/act_scale)
    elif mode == 'perchannel':
        if kwargs['use_v2']:
            scale = [fscale * s / act_scale for s in wscale]
            conv_x = quan_ops.alignment_perchannel_v2(tf.cast(conv_x, tf.int32), scale, data_format='NHWC')
        else:
            conv_x = quan_ops.alignment_perchannel(tf.cast(conv_x, tf.int32), fscale*wscale/act_scale)
    # conv_x = tf.cast(tf.clip_by_value(tf.round(conv_x * fscale * wscale / act_scale), -127, 127), tf.int8)
    if not kwargs['quantize']:
        conv_x = quan_ops.de_quant(conv_x, -act_max, act_max)
    # conv_x = conv_x * fscale * wscale
    return conv_x

def mir_conv2d_bias(inputs, name, strides=(1,1), padding=(0,0,0,0),
        dilation=(1,1), groups=1, channels=None, input_info=None, output_info=None, mode='perlayer', **kwargs):

    x, weight, bias = inputs

    fmax = input_info[0]['max']
    fscale = fmax * 2 / 255

    if mode == 'perlayer':
        wmax = max(input_info[1]['max'])
        wscale = wmax * 2 / 255
    elif mode == 'perchannel':
        wmax = input_info[1]['max']
        wscale = [max(w, 1e-5) * 2 / 255 for w in wmax]
    else:
        raise

    bmax = input_info[2]['max']
    bscale = bmax * 2 / 255

    act_max = output_info[0]['max']
    act_scale = act_max * 2 / 255
    # import pdb; pdb.set_trace()
    # x = tf.round(x / fscale)
    # weight = tf.round(weight / wscale)
    # bias = tf.round(bias / fscale / wscale)
    if not kwargs['input_requantize'][0]:
        # import pdb; pdb.set_trace()
        x = quan_ops.quant(x, -fmax, fmax)

    # import pdb; pdb.set_trace()
    if mode == 'perlayer':
        weight = quan_ops.quant(weight, -wmax, wmax)
    else:
        neg_wmax = [-val for val in wmax]
        weight = quan_ops.quant_perchannel(weight, neg_wmax, wmax, channel_dim_index=3)

    bias = quan_ops.quant(bias, -bmax, bmax)
    u, l, b, r = padding
    if u + b + l + r > 0:
        x = tf.pad(x, [[0,0], [u,b], [l,r], [0,0]])

    conv_x = tf.nn.conv2d(tf.cast(x, tf.float32),
            tf.cast(weight, tf.float32), strides, padding='VALID', dilations=dilation, name=name)
    # if name == '301':
    #     import pdb; pdb.set_trace()
    if kwargs['use_v2']:
        # import pdb; pdb.set_trace()
        if mode == 'perlayer':
            # print(name, bscale/(fscale*wscale))
            bias = quan_ops.alignment_to_high_v2(bias,bscale/(fscale*wscale) )
        elif mode == 'perchannel':
            scale = [bscale / (fscale * s) for s in wscale]
            # print(name, scale)
            bias = quan_ops.alignment_to_high_perchannel_v2(bias, scale, data_format='NHWC')
        else:
            raise
    else:
        if mode == 'perlayer':
            bias = quan_ops.alignment_to_high(bias, bscale/(fscale*wscale) )
        elif mode == 'perchannel':
            scale = [bscale / (fscale * s) for s in wscale]
            bias = quan_ops.alignment_to_high_perchannel(bias, scale, data_format='NHWC')
        else:
            raise
    # bias = tf.cast(bias, tf.float32) * bscale / fscale / wscale
    conv_x = tf.nn.bias_add(conv_x, tf.cast(bias, tf.float32))
    if mode == 'perlayer':
        # import pdb; pdb.set_trace()
        if kwargs['use_v2']:
            conv_x = quan_ops.alignment_v2(tf.cast(conv_x, tf.int32), fscale*wscale/act_scale)
        else:
            conv_x = quan_ops.alignment(tf.cast(conv_x, tf.int32), fscale*wscale/act_scale)
    elif mode == 'perchannel':
        if kwargs['use_v2']:
            scale = [fscale * s / act_scale for s in wscale]
            conv_x = quan_ops.alignment_perchannel_v2(tf.cast(conv_x, tf.int32), scale, data_format='NHWC')
        else:
            conv_x = quan_ops.alignment_perchannel(tf.cast(conv_x, tf.int32), fscale*wscale/act_scale)
    # conv_x = tf.cast(tf.clip_by_value(tf.round(conv_x * fscale * wscale / act_scale), -127, 127), tf.int8)
    if not kwargs['quantize']:
        conv_x = quan_ops.de_quant(conv_x, -act_max, act_max)
    # conv_x = conv_x * fscale * wscale
    return conv_x

def max_pool2d(inputs, name, pool_size=(1,1), strides=(1,1), padding=(0,0),
        input_info=None, output_info=None, **kwargs):
    if len(padding) == 4:
        u, l, b, r = padding
    else:
        u = b = padding[0]
        l = r = padding[1]
    if u + b + l + r > 0:
        input = tf.pad(inputs[0], [[0,0], [u,b], [l,r], [0,0]])
    else:
        input = inputs[0]

    fmax = input_info[0]['max']
    act_max = output_info[0]['max']
    if kwargs['input_requantize'][0]:
        input = quan_ops.de_quant(input, -fmax, fmax)
    x = tf.nn.max_pool2d(input,
            ksize=pool_size, strides=strides, padding='VALID', name=name)
    if kwargs['quantize'] :
        x = quan_ops.quant(x, -act_max, act_max)
    # x = tf.cast(x, tf.int8)
    return x

def mir_add_relu(inputs, name, input_info=None, output_info=None, **kwargs):
    x1, x2 = inputs

    fmax = input_info[0]['max']
    bmax = input_info[1]['max']
    act_max = output_info[0]['max']
    fscale = fmax  * 2/ 255
    bscale = bmax  * 2/ 255
    act_scale = act_max * 2 / 255

    scale = min(fscale, bscale)
    amax = max(fmax, bmax)

    # import pdb; pdb.set_trace()
    if kwargs['input_requantize'][0]:
        # x1 = tf.cast(x1, tf.float32) * fscale / scale
        if fscale / scale > 1.0:
            x1 = tf.cast(quan_ops.alignment_to_high(x1, fscale/scale), tf.float32)
        else:
            x1 = tf.cast(x1, tf.float32)
    else:
        x1 = tf.round(x1 / scale)

    if kwargs['input_requantize'][1]:
        if bscale/scale > 1.0:
            x2 = tf.cast(quan_ops.alignment_to_high(x2, bscale/scale), tf.float32)
        else:
            x2 = tf.cast(x2, tf.float32)
    else:
        x2 = tf.round(x2 / scale)

    x = tf.nn.relu(tf.add(x1, x2))
    if kwargs['use_v2']:
        x = quan_ops.alignment_v2(tf.cast(x, tf.int32), scale/act_scale)
    else:
        x = quan_ops.alignment(tf.cast(x, tf.int32), scale/act_scale)
    if not kwargs['quantize']:
        x = quan_ops.de_quant(x, -act_max, act_max)
    return x

def bias_add(inputs, name, input_info=None, output_info=None, **kwargs):
    x1, x2 = inputs
    fmax = input_info[0]['max']
    bmax = input_info[1]['max']
    act_max = output_info[0]['max']
    fscale = fmax * 2/ 255
    bscale = bmax * 2/ 255
    act_scale = act_max * 2/ 255
    scale = min(fscale, bscale)
    if kwargs['input_requantize'][0]:
        # x1 = tf.cast(x1, tf.float32) * fscale / scale
        if fscale / scale > 1.0:
            x1 = tf.cast(quan_ops.alignment_to_high(x1, fscale/scale), tf.float32)
        else:
            x1 = tf.cast(x1, tf.float32)
    else:
        x1 = tf.round(x1 / scale)

    # x2 = tf.round(x2 / scale)
    x2 = quan_ops.quant(x2, -bmax, bmax)
    if bscale / scale > 1.0:
        x2 = tf.cast(quan_ops.alignment_to_high(x2, bscale/scale), tf.float32)
    else:
        x2 = tf.cast(x2, tf.float32)

    x = tf.add(x1, x2)
    if kwargs['use_v2']:
        x = quan_ops.alignment_v2(tf.cast(x, tf.int32), scale/act_scale)
    else:
        x = quan_ops.alignment(tf.cast(x, tf.int32), scale/act_scale)
    if not kwargs['quantize']:
        x = quan_ops.de_quant(x, -act_max, act_max)
    return x

def global_avg_pool2d(inputs, name, input_info=None, output_info=None, **kwargs):
    fmax = input_info[0]['max']
    act_max = output_info[0]['max']
    if kwargs['input_requantize'][0]:
        x = quan_ops.de_quant(inputs[0], min=-fmax, max=fmax)
    else:
        x = inputs[0]
    x = tf.reduce_mean(x, axis=[1,2], keep_dims=True, name=name)
    if kwargs['quantize']:
        x = quan_ops.quant(x, min=-act_max, max=act_max)
    return x

def batch_flatten(inputs, name, **kwargs):
    return inputs[0]

def dense(inputs, name, input_info=None, output_info=None, **kwargs):
    lhs = input_info[0]['max']
    rhs = input_info[1]['max']
    act_max = output_info[0]['max']
    lscale = lhs * 2/ 255
    rscale = rhs * 2/ 255
    act_scale = act_max * 2/ 255
    if kwargs['input_requantize'][0]:
        lh = tf.cast(inputs[0], tf.float32)
    else:
        lh = tf.round(inputs[0] / lscale)
    rh = tf.cast(quan_ops.quant(inputs[1], -rhs, rhs), tf.float32)
    rst = tf.matmul(lh, rh)
    if kwargs['use_v2']:
        rst = quan_ops.alignment_v2(tf.cast(rst, tf.int32), lscale*rscale/act_scale)
    else:
        rst = quan_ops.alignment(tf.cast(rst, tf.int32), lscale*rscale/act_scale)
    if not kwargs['quantize']:
        rst = quan_ops.de_quant(rst, -act_max, act_max)
    return rst


quantize_function_factory ={
'Const': const,
'mir.conv2d_bias_relu': mir_conv2d_bias_relu,
'mir.conv2d_bias': mir_conv2d_bias,
'nn.max_pool2d': max_pool2d,
'mir.add_relu': mir_add_relu,
'nn.global_avg_pool2d': global_avg_pool2d,
'nn.batch_flatten': batch_flatten,
'nn.dense': dense,
'nn.bias_add': bias_add,
}
