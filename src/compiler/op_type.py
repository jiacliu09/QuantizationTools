from enum import Enum 

class Supported_Op_Type(Enum):
    Add = 'add'
    AdaptiveAvgPool2d = 'nn.adaptive_avg_pool2d'
    AvgPooling = 'nn.avg_pool2d'
    BatchMatmul = 'nn.batch_matmul'
    BatchNorm = 'nn.batch_norm'
    BiasAdd = 'nn.bias_add'
    Cast = 'cast'
    Clip = 'clip' # relu6
    Concat = 'concatenate'
    Const = 'Const'
    Conv2D = 'nn.conv2d' # Depthwise Conv2D set groups, Dilated Conv2D set dilation.
    DeConv2D = 'nn.conv2d_transpose'
    Exp = 'exp'
    ExpandDims = 'expand_dims'
    GlobalAvgPooling = 'nn.global_avg_pool2d' # equal to mean
    Identity = 'Identity'
    LeakyRelu = 'nn.leaky_relu'
    Matmul = 'nn.dense' # fully connected = nn.batch_flatten(optional) + nn.dense + nn.bias_add(optional)
    Max = 'max' # reduce_max
    MaxPooling = 'nn.max_pool2d'
    Mean = 'mean' # reduce_mean
    Multiply = 'multiply' #also square
    Ones = 'ones'
    OnesLike = 'ones_like'
    Padding = 'nn.pad'
    Power = 'power'
    Prelu = 'nn.prelu' # for H-swish
    Relu = 'nn.relu'
    Reshape = 'reshape'
    Sigmoid = 'sigmoid'
    Slice = 'strided_slice'
    SliceLike = 'slice_like'
    Softmax = 'nn.softmax'
    Split = 'split'
    Squeeze = 'squeeze'
    Subtraction = 'subtract'
    Sum = 'sum' # reduce_sum
    Tanh = 'tanh'
    Transpose = 'transpose'
    TopK = 'topk'
    Upsample = 'nn.upsampling'
    Zeros = 'zeros'
    ZerosLike = 'zeros_like'


class Mir_Op_Type(Enum):
    AddRelu = 'mir.add_relu'
    AddReluUpsampling = 'mir.add_relu_upsampling'
    BiasClip = 'mir.bias_clip'
    BiasRelu = 'mir.bias_relu'
    ConvAdd = 'mir.conv2d_add'
    ConvAddRelu = 'mir.conv2d_add_relu'
    ConvBias = 'mir.conv2d_bias'
    ConvBiasAdd = 'mir.conv2d_bias_add'
    ConvBiasAddRelu = 'mir.conv2d_bias_add_relu'
    ConvAddReluUpsampling = 'mir.conv2d_bias_relu_upsampling'
    ConvBiasRelu = 'mir.conv2d_bias_relu'
    ConvBiasClip = 'mir.conv2d_bias_clip'
    Gelu = 'gelu'
    Reciprocal = 'reciprocal'
    ScaleBias = 'mir.scale_bias'
    ScaleBiasAdd = 'mir.scale_bias_add'
    ScaleBiasAddRelu = 'mir.scale_bias_add_relu'
    ScaleBiasRelu = 'mir.scale_bias_relu'

class Might_Support_Op_Type(Enum):
    Argmax = 'argmax'
    Divide = 'divide'
    GatherNd = 'gather_nd'
    Gather = 'take'
    MaxPooling3D = 'nn.max_pool3d'
    Sqrt = 'sqrt'
