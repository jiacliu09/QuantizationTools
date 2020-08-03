import torch
from torch import nn
import numpy as np

class FakeBaseModule(nn.Module):
    def __init__(self, gnode, params):
        super(FakeBaseModule, self).__init__()
        attrs = gnode.get('attrs')
        if attrs is None:  attrs = {}
        inputs = gnode.get('inputs')
        if inputs is None:  inputs = []
        self.attrs = attrs
        self.inputs = inputs
        self.name = gnode.get('name')
        self.params = params
        self.module = nn.Sequential()

    def forward(self, x):
        return self.module(x)

    def load_weights(self):
        pass

    @property
    def nnz(self):
        if isinstance(self.module, (nn.Conv2d, nn.Linear)):
            weight = self.params[self.inputs[1]]
            return np.nonzero(weight)[0].size
        else:
            return 0

    @property
    def sparsity(self):
        if isinstance(self.module, (nn.Conv2d, nn.Linear)):
            weight = self.params[self.inputs[1]]
            return 1 - np.nonzero(weight)[0].size / weight.size
        else:
            return 0.0

    @property
    def real_inputs(self):
        return self.inputs[0:1]

    @property
    def spec_name(self):
        name = self.name + '_' + self.module.__class__.__name__
        return name

class FakeConv2d(FakeBaseModule):

    def __init__(self, gnode, params, biasnode=None):
        super(FakeConv2d, self).__init__(gnode, params)

        in_channel = self.attrs['A_shape'][0][1]
        out_channel = self.attrs['O_shape'][0][1]
        strides = self.attrs['strides']
        padding = self.attrs['padding']
        groups = self.attrs['groups']
        dilation = tuple(self.attrs['dilation'])
        if len(padding) == 4: padding = padding[:2]
        kernel_size = self.attrs['kernel_size']
        self.biasnode = biasnode
        self.module = nn.Conv2d(in_channel, out_channel, kernel_size, stride=strides, padding=padding, bias=biasnode is not None, groups=groups, dilation=dilation)

    def load_weights(self):
        weight = self.params[self.inputs[1]]
        self.module.weight.data.copy_(torch.from_numpy(weight))

        if self.biasnode is not None:
            biasinputs = self.biasnode.get('inputs')
            assert biasinputs[0] == self.name
            bias = self.params[biasinputs[1]]
            self.module.bias.data.copy_(torch.from_numpy(bias))

class FakeBatchNorm(FakeBaseModule):
    def __init__(self, gnode, params):
        super(FakeBatchNorm, self).__init__(gnode, params)

        eps = self.attrs['epsilon']
        out_channel = in_channel = self.attrs['A_shape'][1][0]
        self.module = nn.BatchNorm2d(in_channel, eps=eps, momentum=0.9)

    def load_weights(self):

        gamma = self.params[self.inputs[1]]
        beta = self.params[self.inputs[2]]
        running_mean = self.params[self.inputs[3]]
        running_var = self.params[self.inputs[4]]
        self.module.weight.data.copy_(torch.from_numpy(gamma))
        self.module.bias.data.copy_(torch.from_numpy(beta))
        self.module.running_mean.data.copy_(torch.from_numpy(running_mean))
        self.module.running_var.data.copy_(torch.from_numpy(running_var))

class FakeReLU(FakeBaseModule):
    def __init__(self, gnode, params):
        super(FakeReLU, self).__init__(gnode, params)
        self.module = nn.ReLU(inplace=True)

class FakeMaxPool2d(FakeBaseModule):
    def __init__(self, gnode, params):
        super(FakeMaxPool2d, self).__init__(gnode, params)
        self.pool_size = self.attrs.get('pool_size', (1,1))
        self.strides = self.attrs.get('strides', (1,1))
        self.padding = self.attrs.get('padding', (0,0))
        #ceil_mode = self.attrs.get('ceil_mode', 0)
        #ceil_mode = True if ceil_mode else False
        self.module = nn.MaxPool2d(self.pool_size, stride=self.strides,
                padding=0, ceil_mode=False)

    def forward(self, x):
        u, l, b, r = self.padding
        if u + l + b + r > 0:
            p2d = (u,b,l,r)
            x = nn.functional.pad(x, p2d)
        return self.module(x)

class FakeAdd(FakeBaseModule):
    def __init__(self, gnode, params):
        super(FakeAdd, self).__init__(gnode, params)

    def forward(self, x, y):
        return x+y

    @property
    def real_inputs(self):
        return self.inputs

class FakeGlobalAvgPooling(FakeBaseModule):
    def __init__(self, gnode, params):
        super(FakeGlobalAvgPooling, self).__init__(gnode, params)
        self.module = nn.AdaptiveAvgPool2d(1)

class FakeFlatten(FakeBaseModule):
    def __init__(self, gnode, params):
        super(FakeFlatten, self).__init__(gnode, params)
        self.module = nn.Flatten()

class FakeDense(FakeBaseModule):
    def __init__(self, gnode, params, biasnode=None):
        super(FakeDense, self).__init__(gnode, params)
        out_channel = self.attrs['A_shape'][1][0]
        in_channel = self.attrs['A_shape'][1][1]
        self.biasnode = biasnode
        self.module = nn.Linear(in_channel, out_channel, bias=biasnode is not None)

    def load_weights(self):
        weight = self.params[self.inputs[1]]
        self.module.weight.data.copy_(torch.from_numpy(weight))
        if self.biasnode is not None:
            biasinputs = self.biasnode.get('inputs')
            assert biasinputs[0] == self.name
            bias = self.params[biasinputs[1]]
            self.module.bias.data.copy_(torch.from_numpy(bias))

class FakeReLU6(FakeBaseModule):
    def __init__(self, gnode, params):
        super(FakeReLU6, self).__init__(gnode, params)
        self.module = nn.ReLU6(inplace=True)

class FakeAdaptiveAvgPool2d(FakeBaseModule):
    def __init__(self, gnode, params):
        super(FakeAdaptiveAvgPool2d, self).__init__(gnode, params)
        output_size = self.attrs['output_size']
        if len(output_size) == 1:
            output_size = output_size[0]
        self.module = nn.AdaptiveMaxPool2d(output_size)

class FakeSigmoid(FakeBaseModule):
    def __init__(self, node, params):
        super(FakeSigmoid, self).__init__(node, params)
        self.module = nn.Sigmoid()

class FakeUpsample(FakeBaseModule):
    def __init__(self, node, params):
        super(FakeUpsample, self).__init__(node, params)
        scale_h = self.attrs['scale_h']
        scale_w = self.attrs['scale_w']
        self.module = nn.Upsample(scale_factor=(scale_h, scale_w))

class FakeEqual(FakeBaseModule):
    def __init__(self, node, params):
        super(FakeEqual, self).__init__(node, params)

    def forward(self, x, y):
        return x == y

    @property
    def real_inputs(self):
        return self.inputs

class FakeMultiply(FakeBaseModule):
    def __init__(self, node, params):
        super(FakeMultiply, self).__init__(node, params)

    def forward(self, x, y):
        return x * y

    @property
    def real_inputs(self):
        return self.inputs

class FakeCast(FakeBaseModule):
    def __init__(self, node, params):
        super(FakeCast, self).__init__(node, params)
        self.dtype = self.attrs['dtype']

    def forward(self, x):
        if self.dtype == 'float32':
            return x.float()
        else:
            raise

class FakePad(FakeBaseModule):
    def __init__(self, node, params):
        super(FakePad, self).__init__(node, params)
        self.pad_top, self.pad_bot = self.attrs['pad_width'][2]
        self.pad_left, self.pad_rigth = self.attrs['pad_width'][3]


    def forward(self, x):
        return nn.functional.pad(x, (self.pad_left, self.pad_rigth, self.pad_top, self.pad_bot))



module_factory = {
'Const': FakeBaseModule,
'nn.conv2d': FakeConv2d,
'nn.relu': FakeReLU,
'clip': FakeReLU6,
'nn.batch_norm': FakeBatchNorm,
'nn.max_pool2d': FakeMaxPool2d,
'add': FakeAdd,
'nn.global_avg_pool2d': FakeGlobalAvgPooling,
'nn.batch_flatten': FakeFlatten,
'nn.dense': FakeDense,
'nn.bias_add': FakeBaseModule,
'nn.adaptive_avg_pool2d': FakeAdaptiveAvgPool2d,
'sigmoid': FakeSigmoid,
'multiply':FakeMultiply,
'nn.upsampling': FakeUpsample,
'equal': FakeEqual,
'cast': FakeCast,
'nn.pad': FakePad,
}
