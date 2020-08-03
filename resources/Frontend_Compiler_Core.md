[TOC]
# Frontend Compiler Core 
## Compiler

## Reconstructor
### Code struct
![](/uploads/upload_7693e444867c70f17613e98ea6e4134e.png)

### Interface
```
class BaseReconstructor(object):
    def __init__(self, graph, params, input_node_ids=[], output_node_ids=[], update_node_cfg=[]):
        graph: moffett ir graph info
        params: moffett ir params info
        input_node_ids: spec node ids for input, if not set use graph node 0
        output_node_ids: spec node ids for output, if not set use the node inferenced by input_node_ids
        update_node_cfg: update the node attributes as you set
```

### How to use reconstructor
In torch:
```
with open(graph_json_file, 'r') as f:
    graph = json.load(f)
params = np.load(params_npz_file, allow_pickle=True)['arr_0'][()]
trc = src.TorchReconstructor(graph, params) 
trc.load_weights()
```
In tvm:
```
with open(graph_json_file, 'r') as f:
    graph = json.load(f)
params = np.load(params_npz_file, allow_pickle=True)['arr_0'][()]
tvc = src.TVMReconstructor(graph, params)
tvc._execute()
```
In tensorflow:
```
with open(graph_json_file, 'r') as f:
    graph = json.load(f)
params = np.load(params_npz_file, allow_pickle=True)['arr_0'][()]
params = src.transform_weight_from_mxnet_to_tensorflow(params)
trc = src.TFReconstructor(graph, params)
```

## Calibration

### Code struct
![](/uploads/upload_53f2f8b058542eed9c81285a77e7f7c4.png)

### How to use calibration
```
with open(graph_json_file, 'r') as f:
    graph = json.load(f)
params = np.load(params_npz_file, allow_pickle=True)['arr_0'][()]
params = src.transform_weight_from_mxnet_to_tensorflow(params)
trans = [
    ["SwapRGB", {}],
    ["Resize", {"short_edge": 256, "keep_ratio": true}],
    ["CenterCrop", {"size": 224}],
    ["Normalize", {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}]
]
dataset = src.CalibraDataset(iamge_dir, batchsize, transformer=trans)
calibrator = src.quantization.Calibration(graph, params, dataset)
table = calibrator.run()
```


## Quantization
### Code struct
![](/uploads/upload_1fa40a8ffca7461af9700b31bc3e1953.png)

### Interface

```
quantizer = BaseQaun(weight_quantize_strategy, image_path, graph, params, table=mytable)

- weight_quantize_strategy: perlayer or perchannel
- image_path: image for anisys quantization similarity
- graph: moffett ir graph info
- params: moffett ir params info
- table: calibration table results
```

### How to use with moffett ir
```
with open(graph_json_file, 'r') as f:
    graph = json.load(f)
params = np.load(params_npz_file, allow_pickle=True)['arr_0'][()]
params = src.transform_weight_from_mxnet_to_tensorflow(params)

tf_graph = tf.Graph()
quantization_config = dict(
    weight_quan = perlayer,
    table = 'resnet50v1b_calibration.json',
    image_path = 'image_for_compare_results'
)
with tf_graph.as_default():
    quan_instance = src.get_quan('symmerty_max', quantization_config, graph, params)
    quan_instance.excute()
    
tf.io.write_graph(tf_graph, './', 'quantization.pb', as_text=False)
```

