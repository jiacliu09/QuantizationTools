import os, sys
import json
import numpy as np
import argparse

libpath = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(libpath)
import src

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-graph', '--graph-json',
            default='models/resnet50v1b/moffett_ir/IR_fused_for_CModel_graph.json', help='full graph json')
    parser.add_argument('-param', '--params-file',
            default='models/resnet50v1b/moffett_ir/IR_fused_for_CModel_params.npz', help='full params path')
    parser.add_argument('-input', '--image-dir', default='models/images/calibration')
    parser.add_argument('-ppc', '--preprocess-config', default='configs/mxnet_imagenet_trans.json',
            help='json config of preprocess')
    parser.add_argument('-o', '--output', default='calibrations/resnet50_v1b.json')
    parser.add_argument('--use-kl', action='store_true', default=False, help='flag of use kl')
    args = parser.parse_args()

    with open(args.graph_json, 'r') as f:
        json_graph = json.load(f)
    params = np.load(args.params_file, allow_pickle=True)['arr_0'][()]
    params = src.transform_weight_from_mxnet_to_tensorflow(params)
    image_list = src.utils.get_image_list(args.image_dir)

    preprocessor = src.transform.JsonTrans(args.preprocess_config)
    calibrate_dataset = src.CalibDataset(args.image_dir, 25, transformer=preprocessor)
    calibrator = src.quantization.Calibration(json_graph, params, calibrate_dataset)

    # table = calibrator.run()
    if args.use_kl:
        table = calibrator.run_kl()
    else:
        table = calibrator.run()
    with open(args.output, 'w') as f:
        json.dump(table, f, indent=2)
