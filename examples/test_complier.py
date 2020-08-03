import os, sys
import argparse

libpath = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(libpath)
import src

from src.compiler.gen_relay_ir import gen_relay_ir
from src.compiler.visualize import visualize
from src.compiler.compiler import Compiler
from src.compiler.utils import dump_params

def frontendcompile(args):
    platform = args.platform
    model_path = args.model_path
    epoch = args.epoch
    output_path = args.output_path
    dump_params = args.dump_params
    img_shape = args.img_shape
    reconstruct_graph_name = args.reconstruct_graph_name
    reconstruct_params_name = args.reconstruct_params_name
    compiler_params_name = args.compiler_params_name
    compiler_graph_name = args.compiler_graph_name

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    model = gen_relay_ir(platform, model_path, img_shape, epoch)
    graph, params = visualize(model)

    tran_compiler = Compiler(graph, params)

    # Generate Moffett IR for reconstruct
    if reconstruct_graph_name and reconstruct_params_name:
        tran_compiler.save(reconstruct_graph_name, reconstruct_params_name, output_path)

    # Generate operation fused graph.
    if compiler_graph_name and compiler_params_name:
        tran_compiler.compile()
        # tran_compiler.CModel_transforms()
        tran_compiler.save(compiler_graph_name, compiler_params_name, output_path)
        if dump_params:
            dump_params(tran_compiler.params, compiler_params_name, output_path)
    print('Done!')

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(usage="main.py platform model_path [optional]\ni.e.: pyhton3 main.py mxnet resnet50_v1d --epoch 159")
    arg_parser.add_argument("platform", type=str, help='model platform, i.e.: mxnet, tensorflow...')
    arg_parser.add_argument("model_path", type=str, help='path of model file, For mxnet model, prefix of model name\
                                                          (symbol will be loaded from prefix-symbol.json, parameters will be loaded from prefix-epoch.params.)\
                                                          For tensorflow model, *.pb')
    arg_parser.add_argument("--epoch", default = 0, type=int, help='Only used for mxnet model. epoch number of mxnet model we would like to load, default: 0.')
    arg_parser.add_argument("--output_path", default = "./moffett_ir", type=str, help='output directory of Moffett IR graphs and params, default: ./moffett_ir')
    arg_parser.add_argument("--dump_params", default=False, type=bool, help="Whether dump params to separate files.")
    arg_parser.add_argument("--img_shape", default=[224,224], nargs=2, type=int, help="The image shape of input in model, default: 224 224")
    arg_parser.add_argument("--reconstruct_graph_name", default = 'IR_for_reconstruct_graph', type=str, help='file name of Moffett IR for reconstruct graph, default: IR_for_reconstruct_graph')
    arg_parser.add_argument("--reconstruct_params_name", default = 'IR_for_reconstruct_params', type=str, help='file name of Moffett IR for reconstruct params, default: IR_for_reconstruct_params')
    arg_parser.add_argument("--compiler_graph_name", default='IR_fused_for_CModel_graph', type=str, help='file name of Moffett IR fused for CModel graph, default: IR_fused_for_CModel_graph')
    arg_parser.add_argument("--compiler_params_name", default = 'IR_fused_for_CModel_params', type=str, help='file name of Moffett IR fused for CModel params, default: IR_fused_for_CModel_params')
    args = arg_parser.parse_args()

    frontendcompile(args)
