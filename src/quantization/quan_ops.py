from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

quan_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile('./_moffett_quantization_ops.so'))
