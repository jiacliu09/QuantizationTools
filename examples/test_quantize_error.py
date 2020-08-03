import json
import tensorflow as tf
import numpy as np
import cv2
import yaml
import easydict
import argparse
import os, sys
libpath = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(libpath)
import src

os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
os.environ["CUDA_VISIBLE_DEVICES"]='-1'
def run(graph, params, config, evaluation, save_path):
    params = src.transform_weight_from_mxnet_to_tensorflow(params)

    tf_graph = tf.Graph()
    with tf_graph.as_default():
        quan_instance = src.get_quan(config.strategy, config.qconfig, graph, params)
        quan_instance.execute()

    with tf.Session(graph=tf_graph) as sess:
        input_arr = src.load_images(evaluation.input_images)
        # input_arr = src.load_cifar10(evaluation.input_images)
        output_arr = sess.run(quan_instance.node_dict[evaluation.output_node],
                feed_dict={quan_instance.node_dict[evaluation.input_node]: input_arr})
    if save_path:
        tf.io.write_graph(tf_graph, os.path.dirname(save_path), os.path.basename(save_path), as_text=False)
    return output_arr, quan_instance.act_list

def merge_args_to_config(config, args):
    if args.graph_json:
        config.MODEL.graph = args.graph_json
    if args.params_file:
        config.MODEL.params = args.params_file
    if args.strategy:
        config.QUAN.strategy = args.strategy
    if args.weight_quan:
        config.QUAN.qconfig.weight_quan = args.weight_quan

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', default='config.yml', type=str, help='quantization config')
    parser.add_argument('-g', '--graph-json', default='', help='CModel graph json')
    parser.add_argument('-p', '--params-file', default='', help='CModel params path')
    parser.add_argument('-s', '--strategy', default='', help='quantization strategy')
    parser.add_argument('-wq', '--weight-quan', default='', help='per layer or per channel')
    args = parser.parse_args()
    with open(args.config_file, 'r') as  f:
        config = easydict.EasyDict(yaml.load(f.read()))
    merge_args_to_config(config, args)
    with open(config.MODEL.graph, 'r') as f:
        graph = json.load(f)
    params = np.load(config.MODEL.params, allow_pickle=True)['arr_0'][()]

    print(config)
    out_quan, quan_act_list = run(graph, params, config.QUAN, config.EVALUATION, config.SAVE_PATH)
    config.QUAN.strategy = 'null'
    print(config)
    out, act_list = run(graph, params, config.QUAN, config.EVALUATION, '')
    assert len(quan_act_list) == len(act_list)
    for idx in range(len(act_list)):
        try:
            print(act_list[idx]['node'], src.distance(quan_act_list[idx]['act'], act_list[idx]['act']))
        except Exception as err:
            import pdb; pdb.set_trace()
            print(act_list[idx])
    print('final distance', src.distance(out_quan, out))
