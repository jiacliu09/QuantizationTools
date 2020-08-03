import scipy.spatial.distance
import numpy as np
import cv2
import gluoncv
import mxnet as mx
import os
import tensorflow as tf

def find_node(nodes, idx):
    for node in nodes:
        if node['name'] == idx:
            return node
    return None

def get_input_node(node):
    inputs = node.get('inputs', [])
    if inputs is None: inputs = []
    return [id_ for id_ in inputs if not id_.startswith('params/')]

def readlines(filename):
    with open(filename, 'r') as f:
        lines = f.read().strip().split('\n')
    return lines

def transform_weight_from_mxnet_to_tensorflow(params):
    new_params =  {}
    for name in params:
        if params[name].ndim == 4:
            arr = params[name].transpose([2,3,1,0])
        elif params[name].ndim == 2:
            arr = params[name].transpose([1,0])
        elif params[name].ndim == 1:
            arr = params[name]
        elif params[name].ndim == 0:
            arr = params[name]
        else:
            if name.endswith('tracked'):
                continue
            # import pdb; pdb.set_trace()
            raise "Unknown params dim"
        new_params[name] = arr
    return new_params

def get_image_list(image_dir, suffix=None):
    image_list = []
    for root, dirname, filenames in os.walk(image_dir):
        if not filenames:
            continue
        for fn in filenames:
            if suffix is not None:
                if isinstance(suffix, str):
                    suffix = [suffix]
                image_suffix = fn.split('.')[-1]
                if image_suffix not in suffix:
                    continue
            image_list.append(os.path.join(root,fn))
    return image_list

def spec(name, idx=0):
    return name + ':' + str(idx)

def despec(name):
    dename, _ =  name.split(':')
    return dename

def distance(x,y):
    return scipy.spatial.distance.cosine(x.flatten(), y.flatten())

def preprocess_on_coco(imglist, size=(512,512), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    imgs = []
    for img_path in imglist:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(cv2.resize(img, size), cv2.COLOR_BGR2RGB)
        img = (np.float32(img) / 255 - mean) / std
        imgs.append(img)
    return np.array(imgs)

def process_image_batch(image_file_list):
    images = [mx.image.imread(image_file) for image_file in image_file_list]
    images = gluoncv.data.transforms.presets.imagenet.transform_eval(images)
    images = mx.ndarray.concat(*images, dim=0)
    # import pdb; pdb.set_trace()
    images = images.asnumpy().transpose([0,2,3,1])
    return images

def load_images(img_path):
    image_files = []
    for i in os.listdir(img_path):
        if i.endswith(('.jpeg', '.jpg', '.JPEG')):
            image_files.append(os.path.join(img_path, i))
    return process_image_batch(image_files)

def load_cifar10(img_path):
    if isinstance(img_path, str):
        image_list = get_image_list(img_path)
    else:
        image_list = img_path

    imgs = []
    for p in image_list:
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        imgs.append(img)

    imgs = np.array(imgs).astype('float32')
    imgs /= 255.
    imgs = (imgs-0.5) / 0.5
    return imgs

def get_bit_shift_vec(arr):
    vector = []
    for i in range(len(arr)):
        bit_shift = get_bit_shift(arr[i])
        vector.append(bit_shift)
    return np.array(vector)

def get_bit_shift(v):
    max_num = max(v, 1e-5)
    base = 0
    while (max_num < 127. * 2 ** base):
        base -= 1
    return base + 1

def quantize(x, base):
    x = tf.clip_by_value(x, -127.*2.0**base, 127.*2.0**base)
    x = x * 2.0 ** (-base)
    return tf.cast(tf.round(x), tf.qint8)

def symmetry_max_qunatize(x, scale):
    # scale = max * 2 / 255
    threshold = 127.499 * scale
    x = tf.clip_by_value(x, -threshold, threshold)
    return tf.cast(tf.round(x / scale), tf.qint8)

def symmetry_max_dequantize(x, scale):
    return tf.cast(x, tf.float32) * tf.cast(scale, tf.float32)

def symmetry_max_quantize_and_dequantize(x, scale):
    x = symmetry_max_qunatize(x, scale)
    x = symmetry_max_dequantize(x, scale)
    return x

def dequantize(x, base):
    x = tf.cast(x, tf.float32)
    return x * 2.0 ** base

def quantize_and_dequantize(x, base):
    x = quantize(x, base)
    x = dequantize(x, base)
    return x

def quantize_v2(x, amin, amax):
    x = tf.maximum(tf.minimum(x, amax), amin)
    out = (x - amin) * (255. / (amax-amin)) - 128
    return tf.cast(tf.round(out), tf.qint8)

def dequantize_v2(x, amin, amax):
    out = (tf.cast(x, tf.float32) + 128) / (255./(amax-amin)) + amin
    return out

def quantize_and_dequantize_v2(x, amin, amax):
    return dequantize_v2(quantize_v2(x, amin, amax), amin, amax)
