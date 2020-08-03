[TOC]
# Frontend Compiler Examples
## test_complier.py
Usage: main.py platform model_path [optional]
  ```
python3 main.py tensorflow /path/to/your/model/resnet50_v1d.pb  --output_path /path/to/your/file
python3 main.py mxnet      /path/to/your/model/resnet50_v1d     --epoch 159
  ```
**Note:** For `mxnet model`, there are two files prefix-symbol.json and prefix-epoch.params. Set `/path/to/your/mxnet/model/prefix` and `--epoch` to load mxnet model.

    positional arguments:
    platform                    model platform, i.e.: mxnet, tensorflow...<br>
    model_path                  path of model file, For mxnet model, prefix of modelname(symbol will be loaded from prefix-symbol.json, parameters will be loaded from prefix-epoch.params.)
                                For tensorflow model, *.pb

    optional arguments:
    -h, --help                  show this help message and exit<br>
    --epoch                     Only used for mxnet model. epoch number of mxnet model we would like to load, default: 0.
    --output_path               output directory of Moffett IR graphs and params, default: ./moffett_ir
    --reconstruct_graph_name    file name of Moffett IR for reconstruct graph, default: IR_for_reconstruct_graph
    --reconstruct_params_name   file name of Moffett IR for reconstruct params, default: IR_for_reconstruct_params
    --compiler_graph_name       file name of Moffett IR fused for CModel graph, default: IR_fused_for_CModel_graph
    --compiler_params_name      file name of Moffett IR fused for CModel params, default: IR_fused_for_CModel_params

If forntend compilation complete successfully, all Moffett IR graphs and params could be found in output_path(default is "./moffett_ir").

    --moffett_ir
     |  IR_for_reconstruct_graph.json
     |  IR_for_reconstruct_graph.png
     |  IR_for_reconstruct_params.npz
     |  IR_fused_for_CModel_graph.json
     |  IR_fused_for_CModel_graph.png
     |  IR_fused_for_CModel_params.npz
     |  IR_fused_for_quantize_graph.json
     |  IR_fused_for_quantize_graph.png
     |  need_quantize_config.json


## test_quantize_error.py

    calculate the differences layer-wise before and after quantization
    -h, --help            show this help message and exit
    -c CONFIG_FILE, --config-file CONFIG_FILE
                        quantization config
    -g GRAPH_JSON, --graph-json GRAPH_JSON
                        CModel graph json
    -p PARAMS_FILE, --params-file PARAMS_FILE
                        CModel params path
    -s STRATEGY, --strategy STRATEGY
                        quantization strategy
    -wq WEIGHT_QUAN, --weight-quan WEIGHT_QUAN
                        per layer or per channel


## test_imagenet_quantize_acc.py
    evaluate the effect of quantization on top1 using imagenet eval dataset
    -h, --help            show this help message and exit
    -c CONFIG_FILE, --config-file CONFIG_FILE
                        quantization config
    -g GRAPH_JSON, --graph-json GRAPH_JSON
                        CModel graph json
    -p PARAMS_FILE, --params-file PARAMS_FILE
                        CModel params path

## test_tensorflow_recostructor.py
    demo the usage of tensorflow reconstructor for inference
    -h, --help            show this help message and exit
    -g GRAPH, --graph GRAPH
                        graph json
    -p PARAMS, --params PARAMS
                        params file
    -i INPUT_NPY, --input_npy INPUT_NPY
                        input numpy or none, when input is none, random input
    -o RESULT_NPY, --result_npy RESULT_NPY
                        result numpy to compare reconstructor result
    -s SAVE_PATH, --save_path SAVE_PATH
                        save pb path
    --post POST           postprocess key
    --shape SHAPE

## test_tensorflow_recostructor_train.py
    demo the usage of tensorflow reconstructor for training
    -h, --help            show this help message and exit
    -g GRAPH, --graph GRAPH
                        graph json
    -p PARAMS, --params PARAMS
                        params file
## test_pytorch_reconstructor.py
    demo the usage of pytorch reconstructor for training
    -h, --help            show this help message and exit
    -g GRAPH, --graph GRAPH
                        graph json
    -p PARAMS, --params PARAMS
                        params file
    -i INPUT_NPY, --input_npy INPUT_NPY
                        input numpy or none, when input is none, random input
    -o RESULT_NPY, --result_npy RESULT_NPY
                        result numpy to compare reconstructor result

## test_gluoncv_centernet_rt.py
    demo the convertion of mxnet detection model to tensorflow version
    -h, --help            show this help message and exit
    -g GRAPH, --graph GRAPH
                        graph json
    -p PARAMS, --params PARAMS
                        params file
    -i IMAGE, --image IMAGE
                        test case
    -o OUTPUT, --output OUTPUT
                        test case output

## test_model_complexity.py
    -h, --help            show this help message and exit
    -g GRAPH_JSON, --graph-json GRAPH_JSON
                        full graph json
    -p PARAMS_FILE, --params-file PARAMS_FILE
                        full params path
    结果格式： nnz: 3522685, sparsity: 0.8618712639560533, dense_flops: 3948251920.0, sparse_flops: 357813835.0