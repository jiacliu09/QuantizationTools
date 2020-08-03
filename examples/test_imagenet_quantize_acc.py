import tensorflow as tf
import argparse
import easydict
import yaml
import json
import numpy as np
import os, sys
from tqdm import tqdm
import glog
from copy import deepcopy

libpath = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(libpath)
import src
# glog.setLevel(glog.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

def get_label(image_list):
    def _label(image_path):
        return image_path.split('/')[-2]
    return np.array([_label(p) for p in image_list])

def merge_args_to_config(config, args):
    if args.graph_json:
        config.MODEL.graph = args.graph_json
    if args.params_file:
        config.MODEL.params = args.params_file

def get_graph(graph, params, config):
    params = src.transform_weight_from_mxnet_to_tensorflow(params)
    tf_graph = tf.Graph()
    with tf_graph.as_default():
        quan_instance = src.get_quan(
                config.QUAN.strategy,
                config.QUAN.qconfig,
                graph, params)
        quan_instance.execute()
    input_tensor = quan_instance.node_dict[config.EVALUATION.input_node]
    output_tensor = quan_instance.node_dict[config.EVALUATION.output_node]
    return tf_graph, input_tensor, output_tensor

def evalate(graph_json, params, config):
    label_dict = src.readlines(config.EVALUATION.label_file)
    image_list = src.readlines(config.EVALUATION.image_file)
    batch_size = 50
    image_batches = [image_list[i:i+batch_size] for i in range(
        0, len(image_list), batch_size)]
    t_cnt, a_cnt = 0., 0.
    graph, input_tensor, output_tensor = get_graph(json_graph, params, config)

    with tf.Session(graph=graph) as sess:
        for image_files_batch in tqdm(image_batches):
            image_batch = src.process_image_batch(image_files_batch)
            label_batch = get_label(image_files_batch)
            probs = sess.run(output_tensor, feed_dict={input_tensor: image_batch})
            probs = probs.reshape(-1, 1000)
            predictions = np.array(label_dict)[probs.argmax(axis=1)]
            true_cnt = (predictions==label_batch).sum()
            t_cnt += true_cnt
            a_cnt += len(image_files_batch)
    return t_cnt/a_cnt

def test_quantizations(json_graph, params, config, model_name):
    results = {model_name: []}
    config.QUAN.strategy = 'null'
    acc = evalate(json_graph, params, config)
    config.acc = acc
    results[model_name].append(deepcopy(config))
    print(results)

    # config.QUAN.strategy = 'scale_shift'
    # config.QUAN.qconfig.weight_quan = 'perlayer'
    # acc = evalate(json_graph, params, config)
    # config.acc = acc
    # results[model_name].append(deepcopy(config))

    # config.QUAN.strategy = 'scale_shift'
    # config.QUAN.qconfig.weight_quan = 'perchannel'
    # acc = evalate(json_graph, params, config)
    # config.acc = acc
    # results[model_name].append(deepcopy(config))

    # config.QUAN.strategy = 'minmax'
    # config.QUAN.qconfig.weight_quan = 'perlayer'
    # acc = evalate(json_graph, params, config)
    # config.acc = acc
    # results[model_name].append(deepcopy(config))

    # config.QUAN.strategy = 'minmax'
    # config.QUAN.qconfig.weight_quan = 'perchannel'
    # acc = evalate(json_graph, params, config)
    # config.acc = acc
    # results[model_name].append(deepcopy(config))

    # config.QUAN.strategy = 'minmax_t'
    # config.QUAN.qconfig.weight_quan = 'perchannel'
    # acc = evalate(json_graph, params, config)
    # config.acc = acc
    # results[model_name].append(deepcopy(config))

    config.QUAN.strategy = 'symmetry_max'
    config.QUAN.qconfig.weight_quan = 'perlayer'
    acc = evalate(json_graph, params, config)
    config.acc = acc
    results[model_name].append(deepcopy(config))
    print(results)

    '''
    config.QUAN.strategy = 'symmetry_max_t'
    config.QUAN.qconfig.weight_quan = 'perlayer'
    acc = evalate(json_graph, params, config)
    config.acc = acc
    results[model_name].append(deepcopy(config))
    print(results)
    '''

    # config.QUAN.strategy = 'symmetry_max'
    # config.QUAN.qconfig.weight_quan = 'perchannel'
    # acc = evalate(json_graph, params, config)
    # config.acc = acc
    # results[model_name].append(deepcopy(config))

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', default='resnet50_v1b.yml', type=str, help='quantization config')
    parser.add_argument('-g', '--graph-json', default='', help='CModel graph json')
    parser.add_argument('-p', '--params-file', default='', help='CModel params path')
    args = parser.parse_args()

    with open(args.config_file, 'r') as  f:
        config = easydict.EasyDict(yaml.load(f.read()))
        # config.QUAN.qconfig.remove_percentage = 0.0001

    merge_args_to_config(config, args)
    with open(config.MODEL.graph, 'r') as f:
        json_graph = json.load(f)
    params = np.load(config.MODEL.params, allow_pickle=True)['arr_0'][()]
    model_name = os.path.basename(args.config_file).split('.')[0]
    results = test_quantizations(json_graph, params, config, model_name)
    with open(model_name + '_result.json', 'w') as f:
        json.dump(results, f, indent=2)

