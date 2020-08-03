import json
import numpy as np
import argparse

import sys, os
import cv2
import tensorflow as tf

libpath = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(libpath)
import src

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--graph', default='models/centernet/IR_fused_for_CModel_graph.json', help='graph json')
    parser.add_argument('-p', '--params', default='models/centernet/IR_fused_for_CModel_params.npz', help='params file')
    parser.add_argument('-i', '--dataset_root', default='/hdd1/data/coco/', help='coco root directory')
    return parser.parse_args()

def coco80_to_coco91_class():
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x

def preprocess(img, size=(512,512), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    img = cv2.cvtColor(cv2.resize(img, size), cv2.COLOR_BGR2RGB)
    img = (np.float32(img) / 255 - mean) / std
    img = img[np.newaxis, :, :, :]
    return img

def evaluate_quantization(graph, output_tensor, gt, labelmap):
    jdict = []
    sess = tf.Session(graph=graph)
    input_tensor = graph.get_tensor_by_name('0:0')
    img_ids = []
    for image_info in tqdm(gt.dataset['images'][:100]):
        file_name = image_info['file_name']
        img_id = image_info['id']
        file_path = os.path.join(val_dir, file_name)
        img = cv2.imread(file_path)
        h, w = img.shape[:2]
        img_in = preprocess(img, (512,512))
        result = sess.run(output_tensor, feed_dict={input_tensor: img_in})
        boxes = result[0][:, : 4 ] * [w, h, w, h]
        scores = result[0][:, 4]
        cats = [labelmap[int(id_)] for id_ in result[0][:, 5]]
        for bbox, score, cat in zip(boxes, scores, cats):
            jdict.append({
                    'image_id': img_id,
                    'category_id': cat,
                    'bbox': bbox.tolist(),
                    'score': float(score),
                })
        img_ids.append(img_id)
    sess.close()
    with open('temp_results.json', 'w') as f:
        json.dump(jdict, f)
    dt = gt.loadRes('temp_results.json')
    ev = COCOeval(gt, dt, 'bbox')
    ev.params.imgIds = img_ids
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

if __name__ == '__main__':

    # tf_graph = tf.get_default_graph()
    # with tf_graph.as_default():
    #     trc = src.TFReconstructor(graph, params)
    #     trc.set_postprocessor('tf_centernet_postprocess')
    #     trc._execute()
    #     output = tf_graph.get_tensor_by_name('post:0')
    args = parse_args()
    val_dir = os.path.join(args.dataset_root, 'val2017')
    anno_file =  os.path.join(args.dataset_root, 'annotations', 'instances_val2017.json')
    gt = COCO(anno_file)
    labelmap = coco80_to_coco91_class()

    with open(args.graph, 'r') as f:
        graph = json.load(f)
    params = np.load(args.params, allow_pickle=True)['arr_0'][()]
    params = src.transform_weight_from_mxnet_to_tensorflow(params)
    spec = lambda x, i : x + ':' + str(i)
    calibration_config = 'configs/mxnet_centernet_trans.json'
    image_dir = 'data/coco/calibration'
    calibration_table = 'calibrations/centernet.json'
    if os.path.exists(calibration_table):
        with open(calibration_table, 'r') as f:
            table =  json.load(f)
    else:
        preprocessor = src.transform.JsonTrans(calibration_config)
        calibrate_dataset = src.CalibDataset(image_dir, 10, transformer=preprocessor)
        calibrator = src.quantization.Calibration(graph, params, calibrate_dataset)
        table = calibrator.run()
        with open(calibration_table, 'w') as f:
            json.dump(table, f, indent=2)
    tf.reset_default_graph()

    tf_graph = tf.Graph()
    with tf_graph.as_default():
        quan_ins = src.quantization.SymmetryMaxQuan('perlayer', image_dir, graph, params, table)
        quan_ins.execute()
        heatmaps = [quan_ins.node_dict[spec(name, 0)] for name in quan_ins.output_node_ids]
        output = src.postprocess.tf_centernet_postprocess(*heatmaps, kth=100)

    tf.io.write_graph(tf_graph, '.', 'centernet_quan.pb', as_text=False)
    evaluate_quantization(tf_graph, output, gt, labelmap)
    print('Done')
