import json
import torch
import numpy as np
import argparse
import sys, os
import tensorflow as tf

libpath = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(libpath)
import src

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--graph', default='./models/sparse_resnet50v1b/IR_for_reconstruct_graph.json', help='graph json')
    parser.add_argument('-p', '--params', default='./models/sparse_resnet50v1b/IR_for_reconstruct_params.npz', help='params file')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    with open(args.graph, 'r') as f:
        graph = json.load(f)
    params = np.load(args.params, allow_pickle=True)['arr_0'][()]
    params = src.transform_weight_from_mxnet_to_tensorflow(params)

    dummpy_input = np.random.randn(1,224,224,3)

    g = tf.Graph()
    with g.as_default():
        trc_inference = src.TFReconstructor(graph, params)
        result = trc_inference(dummpy_input)

    new_g = tf.Graph()
    with new_g.as_default():
        trc = src.TFReconstructorTrain(graph, params)
        trc._execute()
        with tf.variable_scope('test', reuse=tf.AUTO_REUSE):
            featmap = trc.model(is_training=True)
            featmap_test = trc.model(is_training=False)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        # frozen_graph_def = trc.frozen()
        # tf.import_graph_def(frozen_graph_def, name='')
        # tf.io.write_graph(frozen_graph_def, os.path.dirname(args.save_path),
        #         os.path.basename(args.save_path), as_text=True)
        input_tsr = new_g.get_tensor_by_name('test/0:0')
        new_train_result = sess.run(featmap, feed_dict={input_tsr: dummpy_input})
        print(src.distance(result, new_train_result['506']))

        input_tsr = new_g.get_tensor_by_name('test/0_1:0')
        new_test_result = sess.run(featmap_test, feed_dict={input_tsr: dummpy_input})
        print(src.distance(result, new_test_result['506']))

