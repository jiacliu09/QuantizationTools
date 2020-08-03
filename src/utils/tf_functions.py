import tensorflow as tf
import numpy as np

def conv2d(inputs, name, strides=(1,1), padding=(0,0,0,0),
        dilation=(1,1), groups=1, channels=None, **kwargs):
    u, l, b, r = padding
    if u + b + l + r > 0:
        input = tf.pad(inputs[0], [[0,0], [u,b], [l,r], [0,0]])
    else:
        input = inputs[0]
    if (channels is not None) and (groups == channels[0]):
        strides = [1] + strides + [1]
        depthwise_filter = tf.transpose(inputs[1], [0,1,3,2])
        x = tf.compat.v1.nn.depthwise_conv2d(input, depthwise_filter, strides=strides, padding='VALID', dilations=dilation, name=name)
    else:
        x = tf.nn.conv2d(input, inputs[1], strides, padding='VALID', dilations=dilation, name=name)
    return x

def conv2d_train(inputs, name, strides=(1,1), padding=(0,0,0,0),
        dilation=(1,1), groups=1, channels=None, is_training=None, **kwargs):

    input, (np_weight, np_mask) = inputs
    u, l, b, r = padding
    if u + b + l + r > 0:
        input = tf.pad(input, [[0,0], [u,b], [l,r], [0,0]])

    weight = tf.get_variable(name + "_weight", np_weight.shape, trainable=True,
            initializer=tf.initializers.constant(np_weight))
    mask = tf.get_variable(name + "_mask", np_mask.shape, trainable=False,
            initializer=tf.initializers.constant(np_mask))
    masked_weight = weight * mask

    if (channels is not None) and (groups == channels[0]):
        strides = [1] + strides + [1]
        depthwise_filter = tf.transpose(masked_weight, [0,1,3,2])
        x = tf.compat.v1.nn.depthwise_conv2d(input, depthwise_filter, strides=strides, padding='VALID', dilations=dilation, name=name)
    else:
        x = tf.nn.conv2d(input, masked_weight, strides, padding='VALID', dilations=dilation, name=name)
    return x

def batch_norm(inputs, name, epsilon=1e-5, **kwargs):
    gamma, beta, mean, var = inputs[1:]
    x, _, _= tf.nn.fused_batch_norm(inputs[0], gamma, beta, mean, var,
            epsilon=epsilon, is_training=False, name=name)
    return x

def batch_norm_train(inputs, name, epsilon=1e-5, is_training=None, **kwargs):
    # gamma = tf.Variable(inputs[1][0], trainable=True, dtype=tf.float32)
    # beta = tf.Variable(inputs[2][0], trainable=True, dtype=tf.float32)
    # mean = tf.Variable(inputs[3][0], trainable=True, dtype=tf.float32)
    # var = tf.Variable(inputs[4][0], trainable=True, dtype=tf.float32)
    # if is_training:
    #     x, _, _ = tf.nn.fused_batch_norm(inputs[0], gamma, beta, None, None, epsilon=epsilon, is_training=True, name=name)
    # else:
    #     x, _, _ = tf.nn.fused_batch_norm(inputs[0], gamma, beta, mean, var, epsilon=epsilon, is_training=False, name=name)
    # return x
    return tf.layers.batch_normalization(inputs[0],
            gamma_initializer=tf.initializers.constant(inputs[1][0]),
            beta_initializer=tf.initializers.constant(inputs[2][0]),
            moving_mean_initializer=tf.initializers.constant(inputs[3][0]),
            moving_variance_initializer=tf.initializers.constant(inputs[4][0]),
            epsilon=epsilon, training=is_training, trainable=True, name=name)


def relu(inputs, name, **kwargs):
    return tf.nn.relu(inputs[0], name=name)

def max_pool2d(inputs, name, pool_size=(1,1), strides=(1,1), padding=(0,0), **kwargs):
    if len(padding) == 4:
        u, l, b, r = padding
    else:
        u = b = padding[0]
        l = r = padding[1]
    if u + b + l + r > 0:
        input = tf.pad(inputs[0], [[0,0], [u,b], [l,r], [0,0]])
    else:
        input = inputs[0]
    x = tf.nn.max_pool2d(input, ksize=pool_size, strides=strides, padding='VALID', name=name)
    return x

def avg_pool2d(inputs, name, pool_size=(1,1), strides=(1,1), padding=(0,0), **kwargs):
    if len(padding) == 4:
        u, l, b, r = padding
    else:
        u = b = padding[0]
        l = r = padding[1]
    if u + l + b + r > 0:
        input = tf.pad(inputs[0], [[0,0], [u,b], [l,r], [0,0]])
    else:
        input = inputs[0]
    x = tf.nn.avg_pool2d(input, ksize=pool_size, strides=strides, padding='VALID', name=name)
    return x

def clip(inputs, name, a_min=0, a_max=6, **kwargs):
    return tf.clip_by_value(*inputs, a_min, a_max, name=name)

def deconv(inputs, name, padding=(0,0,0,0), strides=(2,2), **kwargs):
    inshape = tf.shape(inputs[0])
    N = inshape[0]
    H = inshape[1]
    W = inshape[2]
    # C = inshape[3]
    out_c = tf.shape(inputs[1])[2]
    output_shape = tf.stack([N, H*2, W*2, out_c], axis=0)
    # N = inputs[0].shape[0].value
    # H = inputs[0].shape[1].value
    # W = inputs[0].shape[2].value
    # out_c = inputs[1].shape[2].value
    # output_shape = tf.constant([N, H*2, W*2, out_c])
    x =  tf.nn.conv2d_transpose(inputs[0], inputs[1], output_shape, strides=strides, padding='SAME', name=name)
    # x = tf.image.resize(inputs[0], (H*2, W*2), name=name)
    return x

def add(inputs, name, **kwargs):
    return tf.add(*inputs, name=name)

def bias_add(inputs, name, **kwargs):
    return tf.nn.bias_add(*inputs, name=name)

def bias_add_train(inputs, name, **kwargs):
    input, (np_bias, _) = inputs
    bias = tf.Variable(np_bias, trainable=True, dtype=tf.float32, name=name+'_bias')
    return tf.nn.bias_add(input, bias, name=name)

def sigmoid(inputs, name, **kwargs):
    return tf.math.sigmoid(inputs[0], name=name)

def equal(inputs, name):
    return tf.math.equal(inputs[0], inputs[1],  name=name)

def cast(inputs, name, dtype=tf.float32, **kwargs):
    return tf.cast(inputs[0], dtype, name=name)

def multiply(inputs, name, **kwargs):
    return tf.math.multiply(inputs[0], inputs[1], name=name)

def topk(inputs, name):
    values, indices =  tf.math.top_k(inputs[0], name=name)
    return values, indices

def shape_of(inputs, name):
    return tf.shape(inputs[0], name=name)

def global_avg_pool2d(inputs, name, **kwargs):
    return tf.reduce_mean(*inputs, axis=[1,2], keep_dims=True, name=name)

def batch_flatten(inputs, name, **kwargs):
    return inputs[0]

def dense_train(inputs, name, units=1000, **kwargs):
    if isinstance(units, (tuple, list)):
        units = units[0]
    x = tf.layers.dense(
            inputs[0], units, use_bias=False, trainable=True,
            kernel_initializer=tf.initializers.constant(inputs[1][0]),
            name=name)
    return x

def dense(inputs, name, **kwargs):
    # import pdb; pdb.set_trace()
    x, weight = inputs
    # weight = tf.transpose(weight, (1,0))
    return tf.matmul(x, weight, name=name)

def const(inputs, name, shape=(1,3,224,224), **kwargs) :
    b, c, h, w = shape
    tx = tf.placeholder(dtype=tf.float32, shape=(None, h, w, c), name=name)
    return tx

def mir_conv2d_bias_relu(inputs, name, strides=(1,1), padding=(0,0,0,0), dilation=(1,1), **kwargs):
    if strides == (1,2) or strides == [1,2]:
        strides = (2,2)
    input, weight, bias = inputs
    with tf.name_scope(name):
        conv_x = conv2d([input, weight], strides=strides, padding=padding, dilation=dilation, **kwargs, name='conv')
        bias_x = bias_add([conv_x, bias], name='bias')
        x = relu([bias_x], name='relu')
    return x

def mir_conv2d_bias(inputs, name, strides=(1,1), padding=(0,0,0,0), dilation=(1,1), **kwargs):

    if strides == (1,2) or strides == [1,2]:
        strides = (2,2)
    input, weight, bias = inputs
    with tf.name_scope(name):
        conv_x = conv2d([input, weight], strides=strides, padding=padding, dilation=dilation, **kwargs, name='conv')
        bias_x = bias_add([conv_x, bias], name='bias')
    return bias_x

def mir_add_relu(inputs, name, **kwargs):
    with tf.name_scope(name):
        x = add(inputs, name='add')
        x = relu([x], name='relu')
    return x

def mir_scale_bias(inputs, name, **kwargs):
    input, scale, bias = inputs
    with tf.name_scope(name):
        x = multiply([input, scale], name='scale')
        x = bias_add([x, bias], name='bias')
    return x

def mir_scale_bias_relu(inputs, name, **kwargs):
    input, scale, bias = inputs
    with tf.name_scope(name):
        x = multiply([input, scale], name='scale')
        x = bias_add([x, bias], name='bias')
        x = relu([x], name='relu')
    return x

def deconv(inputs, name, padding=(0,0,0,0), strides=(2,2), **kwargs):
    inshape = tf.shape(inputs[0])
    N = inshape[0]
    H = inshape[1]
    W = inshape[2]
    # C = inshape[3]
    out_c = tf.shape(inputs[1])[2]
    output_shape = tf.stack([N, H*2, W*2, out_c], axis=0)
    # N = inputs[0].shape[0].value
    # H = inputs[0].shape[1].value
    # W = inputs[0].shape[2].value
    # out_c = inputs[1].shape[2].value
    # output_shape = tf.constant([N, H*2, W*2, out_c])
    x =  tf.nn.conv2d_transpose(
            inputs[0], inputs[1], output_shape, strides=strides, padding='SAME', name=name)
    # x = tf.image.resize(inputs[0], (H*2, W*2), name=name)
    return x

def equal(inputs, name, **kwargs):
    return tf.math.equal(inputs[0], inputs[1],  name=name)

def cast(inputs, name, dtype=tf.float32, **kwargs):
    return tf.cast(inputs[0], dtype, name=name)

def divide(inputs, name, **kwargs):
    return tf.math.divide(*inputs, name=name)

def dropout(inputs, name, rate=0.5, **kwargs):
    return inputs[0]

def softmax(inputs, name, axis=1, **kwargs):
    return tf.nn.softmax(inputs[0], axis=axis, name=name)

def split(inputs, name, indices_or_sections=None, axis=1, **kwargs):
    return tf.split(inputs[0], indices_or_sections[0], axis=3, name=name)

def exp(inputs, name, **kwargs):
    return tf.math.exp(*inputs, name=name)

def tfsum(inputs, name, axis=1, keepdims=False, **kwargs):
    if keepdims == 1:
        keepdims = True
    elif keepdims == 0:
        keepdims = False
    return tf.math.reduce_sum(inputs[0], axis=-1, keepdims=True, name=name)

def reshape(inputs, name, newshape=None, **kwargs):
    return tf.reshape(inputs[0], newshape, name=name)

def reciprocal(inputs, name, **kwargs):
    return tf.reciprocal(inputs[0], name=name)

def identity(inputs, name, index=0, **kwargs):
    return inputs[0][index]

def expand_dims(inputs, name, axis=None, num_newaxis=None, **kwargs):
    assert num_newaxis == 1
    return tf.expand_dims(inputs[0], axis=axis, name=name)

def gather(inputs, name, axis=None, **kwargs):
    return tf.gather(inputs[0], inputs[1], axis=axis[0], name=name)

def one_hot(inputs, name, axis=None, depth=None, dtype=None, **kwargs):
    return tf.one_hot(inputs[0], depth, inputs[1], inputs[2], axis=axis, dtype=dtype, name=name)

def gelu(inputs, name, **kwargs):
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (inputs[0] + 0.044715 * tf.pow(inputs[0], 3)))))
    return inputs[0] * cdf

def strided_slice(inputs, name, begin=None, end=None, strides=None, **kwargs):
    assert len(begin) == len(end)
    if strides == []:
        strides = [1 for _ in range(len(begin))]
    return tf.strided_slice(inputs[0], begin, end, strides=strides, name=name)

def mean(inputs, name, axis=None, exclude=None, keepdims=None, **kwargs):
    return tf.math.reduce_mean(inputs[0], keepdims=bool(keepdims),
            axis=axis[0], name=name)

def subtract(inputs, name, **kwargs):
    return inputs[0] - inputs[1]

def power(inputs, name, **kwargs):
    return tf.math.pow(inputs[0], inputs[1])

def transpose(inputs, name, axes=None, **kwargs):
    return tf.transpose(inputs[0], axes, name=name)

def batch_matmul(inputs, name, **kwargs):
    return tf.linalg.matmul(inputs[0], inputs[1], transpose_b=True, name=name)


def squeeze(inputs, name, axis=None, **kwargs):
    return tf.squeeze(inputs[0],  axis=axis, name=name)

def tanh(inputs, name, **kwargs):
    return tf.math.tanh(*inputs, name=name)

function_factory = {
"Const": const,
"mir.conv2d_bias_relu": mir_conv2d_bias_relu,
"mir.conv2d_bias": mir_conv2d_bias,
'nn.avg_pool2d': avg_pool2d,
'nn.max_pool2d': max_pool2d,
'add': add,
'nn.relu': relu,
'nn.global_avg_pool2d': global_avg_pool2d,
'nn.adaptive_avg_pool2d': global_avg_pool2d,
'nn.batch_flatten': batch_flatten,
'nn.dense': dense,
'nn.dense_train': dense_train,
'nn.bias_add': bias_add,
'nn.bias_add_train': bias_add_train,
'mir.add_relu': mir_add_relu,
'clip': clip,
'nn.conv2d': conv2d,
'nn.conv2d_train': conv2d_train,
'mir.scale_bias': mir_scale_bias,
'mir.scale_bias_relu': mir_scale_bias_relu,
'multiply': multiply,
'nn.batch_norm': batch_norm,
'nn.batch_norm_train': batch_norm_train,
'nn.conv2d_transpose': deconv,
'sigmoid': sigmoid,
'equal': equal,
'cast': cast,
'divide': divide,
'nn.dropout': dropout,
'nn.softmax': softmax,
'split': split,
'exp': exp,
'sum': tfsum,
'reshape': reshape,
'reciprocal': reciprocal,
'1/x': reciprocal,
'Identity': identity,
'expand_dims': expand_dims,
'take': gather,
'one_hot': one_hot,
'strided_slice': strided_slice,
'gelu': gelu,
'mean':  mean,
'subtract': subtract,
'power': power,
'transpose': transpose,
'nn.batch_matmul': batch_matmul,
'squeeze': squeeze,
'tanh': tanh,
}
