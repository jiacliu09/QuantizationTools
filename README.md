This repo provides various tools for neural network front-end compiler, quantization and framework converter among tensorflow, pytorch and mxnet

An example of a resnet50 model from mxnet is used to demo the above mentioned tools

---
### 1. Get Environment Docker

Pull Image from Dockerhub

```
docker pull xhxian/frontendcompiler:0.0.1

docker run -ti -v /path/to/your/file/directory:/directory_name xhxian/frontendcompiler:0.0.1 /bin/bash
```

---
###  2. Frontend compiler usage

### Directory list
```
├── calibrations
├── configs
├── examples
├── models
│   ├── images
│   │   ├── calibration
│   │   └── test
│   └── resnet50v1b
├── projects
│   ├── tensorflow
│   │   └── cifar10
│   │       └── include
│   └── torch
│       └── cifar10
├── resources
│   ├──imagenet_fulltest.list
│   ├──imagenet_lsvrc_2015_synsets.txt
│  
└── src
    ├── compiler
    ├── dataset
    ├── np_ops
    ├── postprocess
    │   └── tf
    ├── quantization
    ├── reconstructor
    └── utils
```

- configs:
```
contain a few configuration files for tools
```

- examples

example codes for using the compiler, quantization and reconstruction tools

More details can be found in [Frontend Compiler Example Documentation](resources/Frontend_Compiler_Examples.md)
```
├── mxnet_eval.py
├── performance_modeling.py
├── test_calibration.py
├── test_coco_quantize_map.py
├── test_complier.py
├── test_gluoncv_centernet_rt.py
├── test_imagenet_quantize_acc.py
├── test_model_complexity_mmcv.py
├── test_model_complexity.py
├── test_pytorch_reconstructor.py
├── test_quantize_error.py
├── test_tensorflow_reconstructor.py
├── test_tensorflow_reconstructor_train.py
└── test_tvm_recontructor.py
```


- projects

projects show two demos: tensorflow reconstruct, pytorch reconstruct
```
|-- tensorflow
|   |-- cifar10
|       |-- include
|-- torch
    |-- cifar10
```

- src

src include the source codes for compiler，quantization，reconstructor，utils. Detailed document can [Frontend Compiler Core Documentation](resources/Frontend_Compiler_Core.md)

```
|-- compiler
|-- dataset
|-- np_ops
|-- postprocess
|   |-- tf
|-- quantization
|-- reconstructor
|-- utils
```
---
### 3. Step-by-step  expample

### step 1: get original model and test images

* We use gluoncv resnet50v1b (top1=77.67 on imagenet) as an example here. [download](https://drive.google.com/drive/folders/1Rz0Z6UQbypHeVxr8lNAwef0n-MVYDZfM?usp=sharing)

* images.tar.gz [download](http://moffett.techx.cn:9001/): imagenet images for quantization calibration and quantization test. Place this image folder under `./models`

* imagenet_eval_sub.tar.gz [download](http://moffett.techx.cn:9001/): One image per class from  the validation dataset of imagenet for testing top1 after quantization. These images should be placed according to `./resources/imagenet_fulltest.list`.    

* imagenet_val.tar.gz [download](http://moffett.techx.cn:9001/): the validation dataset of imagenet for testing top1 after quantization. These images should be placed according to `./resources/imagenet_fulltest.list`.    


### step 2: frontend model compiler test
compile the sparse mxnet model into moffett IR

``` python3 examples/test_complier.py mxnet models/resnet50v1b/resnet50v1b --epoch 0 ```

Moffett IR will be saved in the folder `./moffett_ir`
```
├── IR_for_reconstruct_graph.json
├── IR_for_reconstruct_graph.png
├── IR_for_reconstruct_params.npz
├── IR_fused_for_CModel_graph.json
├── IR_fused_for_CModel_graph.png
└── IR_fused_for_CModel_params.npz
```

### step 3: model complexity
For the given model, calcualte the non-zeros, sparsity, dense flops, sparse flops
```
python3 examples/test_model_complexity.py
        --graph moffett_ir/IR_for_reconstruct_graph.json
        --params moffett_ir/IR_for_reconstruct_params.npz
```
nnz: 25497051, sparsity: 0.00022981689306700304, dense_flops: 3948251920.0, sparse_flops: 3941910018.0

### step 4: reconstructor test
Convert Moffett IR to pytorch, tensorflow models for training and inference, input.npy and result.npy produced by original framework
```
In pytorch:
python3 examples/test_pytorch_reconstructor.py
        --graph moffett_ir/IR_for_reconstruct_graph.json
        --params moffett_ir/IR_for_reconstruct_params.npz
        --input_npy input.npy
        --result_npy result.npy
```

```
In tensorflow:
python3 examples/test_tensorflow_reconstructor.py
        --graph moffett_ir/IR_for_reconstruct_graph.json
        --params moffett_ir/IR_for_reconstruct_params.npz
        --input_npy input.npy
        --result_npy result.npy
        --save_path tf_reconstruct.pb
```

### step 5: Quantization test

Quantization calibration to get the calibration file

```
python3 examples/test_calibration.py 
    -graph moffett_ir/IR_fused_for_CModel_graph.json 
    -param moffett_ir/IR_fused_for_CModel_params.npz 
    -input /path/to/your/calibration/images 
    -ppc configs/mxnet_imagenet_trans.json 
    -o calibrations/resnet50_v1b.json  
    --use-kl
```

Layer-wise comparison of feature maps before and after quantization. The smaller the cosine distance, the better quantization is achieved.

```
python3 examples/test_quantize_error.py --config-file configs/resnet50_v1b.yml
```
The expected output after quantization test:
```
├── moffett_ir
│   ├── IR_for_reconstruct_graph.json
│   ├── IR_for_reconstruct_graph.png
│   ├── IR_for_reconstruct_params.npz
│   ├── IR_fused_for_CModel_graph.json
│   ├── IR_fused_for_CModel_graph.png
│   └── IR_fused_for_CModel_params.npz
├── resnet50v1b-0000.params
├── resnet50v1b_symmetry_max_quantization.pb
└── resnet50v1b-symbol.json
```

An example of the quantization configure file
```
MODEL:
    graph: './models/resnet50_v1b/IR_fused_for_CModel_graph.json'
    params: './models/resnet50_v1b/IR_fused_for_CModel_params.npz'
QUAN:
    strategy: 'symmetry_max' # minmax, null, scale_shift, symmetry_max
    qconfig:
        weight_quan: 'perlayer' # perchannel , perlayer
        table: 'calibrations/resnet50_v1b.json'
        image_path: '/path/to/your/calibration/images'
EVALUATION:
    input_images: '/path/to/your/evaluation/images'
    label_file: 'resources/imagenet_lsvrc_2015_synsets.txt'
    image_file: 'resources/imagenet_fulltest.list'
    input_node: '0:0'
    output_node: '506:0'
SAVE_PATH: 'pbs/resnet50_v1b_quan.pb'
```

### step 6: test quantization accuracy on imagenet dataset 

```
python3 examples/test_imagenet_quantize_acc.py --config-file configs/resnet50_v1b.yml
```
| model |  top 1 before/after quantization |
| --- | --- |
| resnet50 v1b   |  0.7768 / 0.7688      |

