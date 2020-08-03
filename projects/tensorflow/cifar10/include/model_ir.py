from tensorflow.compat import v1 as tf
import sys, json
import numpy as np
sys.path.append('/hdd1/lizhilong/project/frontendcompiler')
from src import TFReconstructorTrain
from src import transform_weight_from_mxnet_to_tensorflow

graph_file = '/hdd1/lizhilong/project/frontendcompiler/projects/cifar10/include/resnet50/IR_for_reconstruct_graph.json'
param_file = '/hdd1/lizhilong/project/frontendcompiler/projects/cifar10/include/resnet50/IR_for_reconstruct_params.npz'

def model(is_training):
    _IMAGE_SIZE = 32
    _IMAGE_CHANNELS = 3
    _NUM_CLASSES = 10


    with open(graph_file, 'r') as f:
        graph = json.load(f)
    params = np.load(param_file, allow_pickle=True)['arr_0'][()]
    params = transform_weight_from_mxnet_to_tensorflow(params)

    with tf.name_scope('main_params'):
        x = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE * _IMAGE_SIZE * _IMAGE_CHANNELS], name='Input')
        y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')
        x_image = tf.reshape(x, [-1, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='images')

        global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

    with tf.variable_scope('resnet50') as scope:
        trc = TFReconstructorTrain(graph, params, output_node_ids=['502'])
        featmap = trc.model(is_training=is_training, input_dict={'0': x_image})

    with tf.variable_scope('fully_connected') as scope:
        flat = tf.reshape(featmap['502'], [-1, 2048])
        softmax = tf.layers.dense(inputs=flat, units=_NUM_CLASSES, name=scope.name)

    y_pred_cls = tf.argmax(softmax, axis=1)

    return x, y, softmax, y_pred_cls, global_step, learning_rate


def lr(epoch):
    learning_rate = 1e-3
    if epoch > 80:
        learning_rate *= 0.5e-3
    elif epoch > 60:
        learning_rate *= 1e-3
    elif epoch > 40:
        learning_rate *= 1e-2
    elif epoch > 20:
        learning_rate *= 1e-1
    return learning_rate
