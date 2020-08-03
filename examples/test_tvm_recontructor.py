import json
import numpy as np
import os
import sys
import tensorflow as tf

libpath = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(libpath)
import src



if __name__ == '__main__':
    with open('bert/IR_fused_for_CModel_graph.json', 'r') as f:
        graph = json.load(f)


    tg = tf.Graph()
    with tg.as_default():
        graph_def = tf.GraphDef()
        with open('/hdd1/xiaohe/model_zoo/Bert-base/bert_model_nomask_L-12-batch1.pb', 'rb') as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

    params = np.load('bert/IR_fused_for_CModel_params.npz', allow_pickle=True)['arr_0'][()]

    input_ids = np.random.randint(0, 1000, (1,128)).astype(np.int32)
    segment_ids = np.random.randint(0, 10, (1,128)).astype(np.int32)


    tvc = src.TVMReconstructor(graph, params, output_node_ids=['1201'])
    tvc._execute()
    rst = tvc(input_ids, segment_ids)

    # trc = src.TFReconstructor(graph, params, output_node_ids=['1201'])
    # trc._execute()
    # rst_tf = trc(input_ids, segment_ids)

    # with tf.Session(graph=tg) as sess:
    #     output_tsr = tg.get_tensor_by_name('bert/pooler/dense/Tanh:0')
    #     # output_tsr = tg.get_tensor_by_name('bert/embeddings/add:0')
    #     input_tsr_1 = tg.get_tensor_by_name('input_ids:0')
    #     input_tsr_2 = tg.get_tensor_by_name('segment_ids:0')
    #     rst_tf = sess.run(output_tsr, feed_dict={input_tsr_1: input_ids, input_tsr_2: segment_ids})


    # import pdb; pdb.set_trace()
    print('Done')
