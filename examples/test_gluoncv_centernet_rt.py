import json
import numpy as np
import argparse
import sys, os
import cv2
import tensorflow as tf

libpath = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(libpath)
import src

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--graph', default='models/centernet/IR_for_reconstruct_graph.json', help='graph json')
    parser.add_argument('-p', '--params', default='models/centernet/IR_for_reconstruct_params.npz', help='params file')

    parser.add_argument('-i', '--image', default='./street.jpg', help='test case')
    parser.add_argument('-o', '--output', default='./result.jpg', help='test case output')
    return parser.parse_args()

def preprocess(img, size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    img = cv2.cvtColor(cv2.resize(img, size), cv2.COLOR_BGR2RGB)
    img = (np.float32(img) / 255 - mean) / std
    img = img[np.newaxis, :, :, :]
    return img

if __name__ == '__main__':
    args = parse_args()
    with open(args.graph, 'r') as f:
        graph = json.load(f)
    params = np.load(args.params, allow_pickle=True)['arr_0'][()]
    params = src.transform_weight_from_mxnet_to_tensorflow(params)

    trc = src.TFReconstructor(graph, params)
    trc.set_postprocessor('tf_centernet_postprocess')
    img = cv2.imread(args.image)
    img_in = preprocess(img, (512, 512))
    result = trc(img_in)

    h, w, _ = img.shape
    boxes = result[0][:, :4] * [w, h, w, h]
    boxes = np.round(boxes).astype(np.int32)
    for i in range(boxes.shape[0]):
        # if result[0][i, 5] < 0.3:
        #     continue
        x, y, w, h = boxes[i]
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 0, 1)
    cv2.imwrite(args.output, img)
    tf.io.write_graph(trc.tf_graph, '.', 'centernet.pb', as_text=False)
