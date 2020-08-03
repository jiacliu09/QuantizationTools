import numpy as np
import logging
import logging.handlers
import os
import json
import math
import argparse
import tvm
import tvm.relay as relay
from tvm.contrib import graph_runtime
from collections import namedtuple
from matplotlib import pyplot as plt
import matplotlib

def calculate_node_timing(op_type, node, formula, conv2d_dict):
    if 'conv2d' in op_type:
        strides = node['attrs']['strides']
        groups = node['attrs']['groups']
        dilation = node['attrs']['dilation']
        padding = node['attrs']['padding']
        layout = node['attrs']['data_layout']
        kernel_layout = node['attrs']['kernel_layout']
        in_shape = node['attrs']['A_shape'][0]
        data = np.ones(in_shape).astype('float32')
        param_shape = node['attrs']['A_shape'][1]
        dense_weight = np.ones(param_shape).astype('float32')

        param_dict = {}
        param_dict['inputs'] = tvm.nd.array(data)
        param_dict['dense_weight'] = tvm.nd.array(dense_weight)
        inputs = relay.var("inputs", shape=in_shape)
        weight = relay.var("dense_weight", shape=param_shape)
        if 'nn.conv2d' == op_type:
            output = tvm.relay.nn.conv2d(inputs, weight, strides=strides, dilation=dilation,
                                    groups=groups, padding=padding, data_layout=layout, out_layout=layout, kernel_layout=kernel_layout)
        elif 'nn.conv2d_tranpose' == op_type:
            output_padding = node['attrs']['output_padding']
            output = tvm.relay.nn.conv2d_transpose(inputs, weight, strides=strides, dilation=dilation,
                    groups=groups, padding=padding, output_padding=output_padding, data_layout=layout, out_layout=layout, kernel_layout=kernel_layout)
        target = 'llvm'
        func = relay.Function(relay.analysis.free_vars(output), output)
        func = relay.build_module.bind_params_by_name(func, param_dict)
        mod = tvm.IRModule()
        mod["main"] = func
        # Build with Relay
        with relay.build_config(opt_level=0): # Currently only support opt_level=0
            graph, lib, params = relay.build(mod, target, params=param_dict)
        # Generate graph runtime
        ctx = tvm.cpu()
        m = graph_runtime.create(graph, lib, ctx)
        m.set_input(**params)
        m.run()
        output_tensor = m.get_output(0).asnumpy()
        FLOPs = np.sum(output_tensor).item()
        W_NNZ = np.sum(dense_weight).item()
        if op_type == 'nn.conv2d_transpose':
            pass
        elif groups == 1 and formula.sparsity:
            in_c = in_shape[1]
            if in_c <= 8:
                FLOPs = FLOPs
                W_NNZ = W_NNZ
            elif in_c <= 16 and in_c > 8:
                FLOPs = FLOPs/2
                W_NNZ = W_NNZ/2
            elif in_c > 16 and in_c <= 32:
                FLOPs = FLOPs/4
                W_NNZ = W_NNZ/4
            elif in_c > 32 and in_c <= 64:
                FLOPs = FLOPs/8
                W_NNZ = W_NNZ/8
            else:
                FLOPs = FLOPs/16
                W_NNZ = W_NNZ/16
        conv2d_dict['total_FLOPs'] += FLOPs
        conv2d_dict['total_WNNZ'] += W_NNZ

        IN_Size = 1
        for item in in_shape:
            IN_Size *= item
        # logging.info('Conv2d FLOPs: {}, W_NNZ {}'.format(FLOPs, W_NNZ))
        logging.info('{} nn.conv2d input shape {} weight shape {}'.format(node['name'], in_shape, param_shape))
        if not formula.f_weight_INT8_cache_flag:
            timing_IO_weight_INT8 = W_NNZ*2/formula.DDR_bandwidth
        else:
            timing_IO_weight_INT8 = W_NNZ*2/formula.W_GLB_Cache_bandwidth
        timing_IO_Halo_INT8 = 0
        if formula.Core2Core_bandwidth:
            timing_IO_weight_INT8 = max(timing_IO_weight_INT8, W_NNZ/formula.Core2Core_bandwidth)
            in_shape_H = in_shape[2]
            in_shape_W = in_shape[3]
            kernel_R = param_shape[2]
            kernel_S = param_shape[3]
            in_c = in_shape[1]
            timing_IO_Halo_INT8 = (in_shape_H/2*(kernel_S-1)/2 + in_shape_W/2*(kernel_R-1)/2 + (kernel_S-1)*(kernel_R-1)/4)*in_c*8/formula.Core2Core_bandwidth
        timing_IO_IN_INT8 = IN_Size/formula.GLB_Bandwidth
        timing_OP_INT8 = FLOPs/formula.PE_Array_MACs + timing_IO_Halo_INT8
        if timing_IO_IN_INT8 > timing_OP_INT8:
            logging.warning('Node {} {} INT8, timing of I/O IN_Size {} > timing of op {}'
                        .format(node['name'], op_type, timing_IO_IN_INT8, timing_OP_INT8))
        elif timing_IO_weight_INT8 > timing_OP_INT8:
            logging.warning('Node {} {} INT8, timing of I/O weight {} > timing of op {}'
                        .format(node['name'], op_type, timing_IO_weight_INT8, timing_OP_INT8))
        elif timing_IO_Halo_INT8 >timing_OP_INT8:
            logging.warning('Node {} {} INT8, timing of I/O Halo {} > timing of op {}'
                        .format(node['name'], op_type, timing_IO_Halo_INT8, timing_OP_INT8))
        timing_INT8 = max(timing_OP_INT8, max(timing_IO_IN_INT8, timing_IO_weight_INT8))
        if timing_INT8 == timing_OP_INT8:
            conv2d_dict['f_op_INT8'].append(node['name'])
        elif timing_INT8 == timing_IO_IN_INT8:
            conv2d_dict['f_I/O_INT8'].append(node['name'])
        elif timing_INT8 == timing_IO_weight_INT8:
            conv2d_dict['f_DDR_INT8'].append(node['name'])
        # logging.info('INT8, f_op {}, f_io_in {}, f_io_w {}, f_Halo {}'.format(timing_OP_INT8, timing_IO_IN_INT8, timing_IO_weight_INT8, timing_IO_Halo_INT8))

        if not formula.f_weight_BF16_cache_flag:
            timing_IO_weight_BF16 = W_NNZ*3/formula.DDR_bandwidth
        else:
            timing_IO_weight_BF16 = W_NNZ*3/formula.W_GLB_Cache_bandwidth
        timing_IO_Halo_BF16 = 0
        if formula.Core2Core_bandwidth:
            timing_IO_weight_BF16 = max(timing_IO_weight_BF16, W_NNZ/formula.Core2Core_bandwidth)
            timing_IO_Halo_BF16 = 2* (in_shape_H/2*(kernel_S-1)/2 + in_shape_W/2*(kernel_R-1)/2 + (kernel_S-1)*(kernel_R-1)/4)*in_c*8/formula.Core2Core_bandwidth
        timing_IO_IN_BF16 = IN_Size/formula.GLB_Bandwidth*2
        timing_OP_BF16 = FLOPs/formula.PE_Array_MACs*2 + timing_IO_Halo_BF16
        if timing_IO_IN_BF16 > timing_OP_BF16:
            logging.warning('Node {} {} BF16, timing of I/O IN_Size {} > timing of op {}'
                        .format(node['name'], op_type, timing_IO_IN_BF16, timing_OP_BF16))
        elif timing_IO_weight_BF16 > timing_OP_BF16:
            logging.warning('Node {} {} BF16, timing of I/O weight {} > timing of op {}'
                        .format(node['name'], op_type, timing_IO_weight_BF16, timing_OP_BF16))
        timing_BF16 = max(timing_OP_BF16, max(timing_IO_IN_BF16, timing_IO_weight_BF16))
        if timing_BF16 == timing_OP_BF16:
            conv2d_dict['f_op_BF16'].append(node['name'])
        elif timing_BF16 == timing_IO_IN_BF16:
            conv2d_dict['f_I/O_BF16'].append(node['name'])
        elif timing_BF16 == timing_IO_weight_BF16:
            conv2d_dict['f_DDR_BF16'].append(node['name'])
        elif timing_BF16 == timing_IO_Halo_BF16:
            conv2d_dict['f_Halo_BF16'].append(node['name'])
        # logging.info('BF16, f_op {}, f_io_in {}, f_io_w {}'.format(timing_OP_BF16, timing_IO_IN_BF16, timing_IO_weight_BF16))
        return timing_INT8, timing_BF16, conv2d_dict
    elif 'nn.dense' == op_type:
        units = node['attrs']['units'][0]
        in_shape = node['attrs']['A_shape'][0]
        data = np.ones(in_shape).astype('float32')
        param_shape = node['attrs']['A_shape'][1]
        dense_weight = np.ones(param_shape).astype('float32')

        param_dict = {}
        param_dict['inputs'] = tvm.nd.array(data)
        param_dict['dense_weight'] = tvm.nd.array(dense_weight)
        inputs = relay.var("inputs", shape=in_shape)
        weight = relay.var("dense_weight", shape=param_shape)
        output = tvm.relay.nn.dense(inputs, weight, units=units)
        target = 'llvm'
        func = relay.Function(relay.analysis.free_vars(output), output)
        func = relay.build_module.bind_params_by_name(func, param_dict)
        mod = tvm.IRModule()
        mod["main"] = func
        # Build with Relay
        with relay.build_config(opt_level=0): # Currently only support opt_level=0
            graph, lib, params = relay.build(mod, target, params=param_dict)
        # Generate graph runtime
        ctx = tvm.cpu()
        m = graph_runtime.create(graph, lib, ctx)
        m.set_input(**params)
        m.run()
        output_tensor = m.get_output(0).asnumpy()
        FLOPs = np.sum(output_tensor).item()
        W_NNZ = np.sum(dense_weight).item()
        if formula.sparsity:
            FLOPs = FLOPs/4
            W_NNZ = W_NNZ/4
        conv2d_dict['total_FLOPs'] += FLOPs
        conv2d_dict['total_WNNZ'] += W_NNZ

        IN_Size = 1
        for item in in_shape:
            IN_Size *= item
        logging.info('{} nn.dense input shape {} weight shape {}'.format(node['name'], in_shape, param_shape))
        if not formula.f_weight_INT8_cache_flag:
            timing_IO_weight_INT8 = W_NNZ*2/formula.DDR_bandwidth
        else:
            timing_IO_weight_INT8 = W_NNZ*2/formula.W_GLB_Cache_bandwidth
        if formula.Core2Core_bandwidth:
            timing_IO_weight_INT8 = max(timing_IO_weight_INT8, W_NNZ/formula.Core2Core_bandwidth)
        timing_IO_IN_INT8 = IN_Size/formula.GLB_Bandwidth
        timing_OP_INT8 = FLOPs/formula.PE_Array_MACs
        if timing_IO_IN_INT8 > timing_OP_INT8:
            logging.warning('Node {} {} INT8, timing of I/O IN_Size {} > timing of op {}'
                        .format(node['name'], op_type, timing_IO_IN_INT8, timing_OP_INT8))
        elif timing_IO_weight_INT8 > timing_OP_INT8:
            logging.warning('Node {} {} INT8, timing of I/O weight {} > timing of op {}'
                        .format(node['name'], op_type, timing_IO_weight_INT8, timing_OP_INT8))
        timing_INT8 = max(timing_OP_INT8, max(timing_IO_IN_INT8, timing_IO_weight_INT8))
        if timing_INT8 == timing_OP_INT8:
            conv2d_dict['f_op_INT8'].append(node['name'])
        elif timing_INT8 == timing_IO_IN_INT8:
            conv2d_dict['f_I/O_INT8'].append(node['name'])
        elif timing_INT8 == timing_IO_weight_INT8:
            conv2d_dict['f_DDR_INT8'].append(node['name'])
        conv2d_dict['PE_Array'].append(timing_OP_INT8)

        if not formula.f_weight_BF16_cache_flag:
            timing_IO_weight_BF16 = W_NNZ*3/formula.DDR_bandwidth
        else:
            timing_IO_weight_BF16 = W_NNZ*3/formula.W_GLB_Cache_bandwidth
        if formula.Core2Core_bandwidth:
            timing_IO_weight_BF16 = max(timing_IO_weight_BF16, W_NNZ/formula.Core2Core_bandwidth)
        timing_IO_IN_BF16 = IN_Size/formula.GLB_Bandwidth*2
        timing_OP_BF16 = FLOPs/formula.PE_Array_MACs*2
        if timing_IO_IN_BF16 > timing_OP_BF16:
            logging.warning('Node {} {} BF16, timing of I/O IN_Size {} > timing of op {}'
                        .format(node['name'], op_type, timing_IO_IN_BF16, timing_OP_BF16))
        elif timing_IO_weight_BF16 > timing_OP_BF16:
            logging.warning('Node {} {} BF16, timing of I/O weight {} > timing of op {}'
                        .format(node['name'], op_type, timing_IO_weight_BF16, timing_OP_BF16))
        timing_BF16 = max(timing_OP_BF16, max(timing_IO_IN_BF16, timing_IO_weight_BF16))
        if timing_BF16 == timing_OP_BF16:
            conv2d_dict['f_op_BF16'].append(node['name'])
        elif timing_BF16 == timing_IO_IN_BF16:
            conv2d_dict['f_I/O_BF16'].append(node['name'])
        elif timing_BF16 == timing_IO_weight_BF16:
            conv2d_dict['f_DDR_BF16'].append(node['name'])

        return timing_INT8, timing_BF16, conv2d_dict
    elif 'nn.batch_matmul' == op_type:
        in0_shape = node['attrs']['A_shape'][0]
        in_0 = np.ones(in0_shape).astype('float32')
        in1_shape = node['attrs']['A_shape'][1]
        in_1 = np.ones(in1_shape).astype('float32')

        param_dict = {}
        param_dict['inputs_0'] = tvm.nd.array(in_0)
        param_dict['inputs_1'] = tvm.nd.array(in_1)
        inputs_0 = relay.var("inputs_0", shape=in0_shape)
        inputs_1 = relay.var("inputs_1", shape=in1_shape)
        output = tvm.relay.nn.batch_matmul(inputs_0, inputs_1)
        target = 'llvm'
        func = relay.Function(relay.analysis.free_vars(output), output)
        func = relay.build_module.bind_params_by_name(func, param_dict)
        mod = tvm.IRModule()
        mod["main"] = func
        # Build with Relay
        with relay.build_config(opt_level=0): # Currently only support opt_level=0
            graph, lib, params = relay.build(mod, target, params=param_dict)
        # Generate graph runtime
        ctx = tvm.cpu()
        m = graph_runtime.create(graph, lib, ctx)
        m.set_input(**params)
        m.run()
        output_tensor = m.get_output(0).asnumpy()
        FLOPs = np.sum(output_tensor).item()
        conv2d_dict['total_FLOPs'] += FLOPs

        IN0_Size = 1
        for item in in0_shape:
            IN0_Size *= item
        IN1_Size = 1
        for item in in1_shape:
            IN1_Size *= item
        timing_IO_IN_INT8 = max(IN0_Size, IN1_Size)/formula.GLB_Bandwidth
        timing_OP_INT8 = FLOPs/formula.PE_Array_MACs
        if timing_IO_IN_INT8 > timing_OP_INT8:
            logging.warning('Node {} {} INT8, timing of I/O IN_Size {} > timing of op {}'
                        .format(node['name'], op_type, timing_IO_IN_INT8, timing_OP_INT8))
        timing_INT8 = max(timing_IO_IN_INT8, timing_OP_INT8)
        if timing_INT8 == timing_OP_INT8:
            conv2d_dict['f_op_INT8'].append(node['name'])
        elif timing_INT8 == timing_IO_IN_INT8:
            conv2d_dict['f_I/O_INT8'].append(node['name'])
        conv2d_dict['PE_Array'].append(timing_OP_INT8)

        timing_IO_IN_BF16 = max(IN0_Size, IN1_Size)/formula.GLB_Bandwidth *2
        timing_OP_BF16 = FLOPs/formula.PE_Array_MACs *2
        if timing_IO_IN_BF16 > timing_OP_BF16:
            logging.warning('Node {} {} BF16, timing of I/O IN_Size {} > timing of op {}'
                        .format(node['name'], op_type, timing_IO_IN_BF16, timing_OP_BF16))
        timing_BF16 = max(timing_IO_IN_BF16, timing_OP_BF16)
        if timing_BF16 == timing_OP_BF16:
            conv2d_dict['f_op_BF16'].append(node['name'])
        elif timing_BF16 == timing_IO_IN_BF16:
            conv2d_dict['f_I/O_BF16'].append(node['name'])

        return timing_INT8, timing_BF16, conv2d_dict
    elif 'nn.upsampling' == op_type:
        OUT_Size = 1
        for item in node['attrs']['O_shape'][0]:
            OUT_Size *= item
        timing_INT8 = OUT_Size/formula.VPU_Bandwidth_INT8
        timing_BF16 = OUT_Size/formula.VPU_Bandwidth_BF16
        return timing_INT8, timing_BF16, conv2d_dict
    elif 'nn.max_pool2d' == op_type or 'nn.avg_pool2d' == op_type:
        IN_Size = 1
        for item in node['attrs']['A_shape'][0]:
            IN_Size *= item
        pool_size = node['attrs']['pool_size'][0]
        strides = node['attrs']['strides'][0]
        timing_INT8 = IN_Size*pool_size/strides/formula.VPU_Bandwidth_INT8
        timing_BF16 = IN_Size*pool_size/strides/formula.VPU_Bandwidth_BF16
        return timing_INT8, timing_BF16, conv2d_dict
    elif 'bias_add' in op_type:
        IN_Size = 1
        for item in node['attrs']['O_shape'][0]:
            IN_Size *= item
        if op_type.startswith('mir'):
            param_shape = node['attrs']['A_shape'][2][0]
        elif op_type.startswith('nn'):
            param_shape = node['attrs']['A_shape'][1][0]
        timing_IO_weight_INT8 = param_shape*2/formula.DDR_bandwidth
        timing_IO_weight_BF16 = param_shape*3/formula.DDR_bandwidth
        timing_OP_INT8 = IN_Size/formula.VPU_Bandwidth_INT8
        timing_OP_BF16 = IN_Size/formula.VPU_Bandwidth_BF16
        timing_INT8 = max(timing_IO_weight_INT8, timing_OP_INT8)
        timing_BF16 = max(timing_IO_weight_BF16, timing_OP_BF16)
        if timing_IO_weight_INT8 > timing_OP_INT8:
            logging.warning('Node {} {} INT8, timing of I/O weight {} > timing of op {}'
                        .format(node['name'], op_type, timing_IO_weight_INT8, timing_OP_INT8))
        if timing_IO_weight_BF16 > timing_OP_BF16:
            logging.warning('Node {} {} BF16, timing of I/O weight {} > timing of op {}'
                        .format(node['name'], op_type, timing_IO_weight_BF16, timing_OP_BF16))
        return timing_INT8, timing_BF16, conv2d_dict
    elif 'nn.relu' == op_type or 'clip' == op_type:
        IN_Size = 1
        for item in node['attrs']['O_shape'][0]:
            IN_Size *= item
        timing_INT8 = IN_Size/formula.VPU_Bandwidth_INT8
        timing_BF16 = IN_Size/formula.VPU_Bandwidth_BF16
        return timing_INT8, timing_BF16, conv2d_dict
    elif 'add' == op_type or 'subtract' == op_type or 'multiply' == op_type:
        IN1_Size = 1
        for item in node['attrs']['A_shape'][0]:
            IN1_Size *= item
        IN2_Size = 1
        for item in node['attrs']['A_shape'][1]:
            IN2_Size *= item
        timing_INT8 = max(IN1_Size, IN2_Size)/formula.VPU_Bandwidth_INT8
        timing_BF16 = max(IN1_Size, IN2_Size)/formula.VPU_Bandwidth_BF16
        conv2d_dict['VPU'].append(timing_INT8)
        return timing_INT8, timing_BF16, conv2d_dict
    elif 'nn.softmax' == op_type:
        IN_Size = 1
        for item in node['attrs']['A_shape'][0]:
            IN_Size *= item
        timing_INT8 = IN_Size/formula.VPU_Bandwidth_INT8
        timing_BF16 = IN_Size/formula.VPU_Bandwidth_BF16
        conv2d_dict['VPU'].append(timing_INT8*4)
        return timing_INT8*4, timing_BF16*4, conv2d_dict
    elif 'expand_dims' == op_type or 'squeeze' == op_type or 'reshape' == op_type:
        timing_INT8 = 0
        timing_BF16 = 0
        conv2d_dict['VPU'].append(timing_INT8)
        return timing_INT8, timing_BF16, conv2d_dict
    elif 'transpose' == op_type:
        IN_Size = 1
        for item in node['attrs']['A_shape'][0]:
            IN_Size *= item
        timing_INT8 = IN_Size/formula.VPU_Bandwidth_INT8
        timing_BF16 = IN_Size/formula.VPU_Bandwidth_BF16
        conv2d_dict['transpose'].append(timing_INT8)
        return timing_INT8, timing_BF16, conv2d_dict
    else:
        IN_Size = 1
        for item in node['attrs']['A_shape'][0]:
            IN_Size *= item
        timing_INT8 = IN_Size/formula.VPU_Bandwidth_INT8
        timing_BF16 = IN_Size/formula.VPU_Bandwidth_BF16
        conv2d_dict['VPU'].append(timing_INT8)
        return timing_INT8, timing_BF16, conv2d_dict

def calculate_timing(args):
    json_file = args.IR_fused_for_CModel_graph
    sparsity = args.sparsity
    log = args.log
    weight_cache = args.weight_cache
    output_path = args.output_path
    with open(json_file, 'r') as f:
            mir_graph = json.load(f)
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    # Generate log file to record theoretical timing
    log_name = os.path.join(output_path, log + '.log')
    if os.path.exists(log_name):
        os.remove(log_name)
    handler = logging.handlers.WatchedFileHandler(os.environ.get("LOGFILE", log_name))
    formatter = logging.Formatter(logging.BASIC_FORMAT)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    root.addHandler(handler)

    op_list = set(['nn.conv2d', 'nn.conv2d_transpose', 'nn.max_pool2d', 'nn.avg_pool2d',
                    'nn.dense', 'nn.upsampling', 'mean', 'max', 'sum',
                    'split', 'concatenate', 'reshape', 'transpose', 'strided_slice',
                    'slice_like', 'nn.leaky_relu', 'nn.prelu', 'exp', 'clip', 'sigmoid',
                    'nn.softmax', 'nn.global_avg_pool2d', 'nn.adaptive_avg_pool2d', 'add',
                    'subtract', 'multiply', 'nn.bias_add', 'nn.relu', 'mir.scale_bias_relu',
                    'mir.conv2d_bias', 'mir.conv2d_bias_relu', 'mir.scale_bias', 'mir.add_relu',
                    'mir.conv2d_bias_clip', 'gelu', 'nn.batch_matmul', 'expand_dims', 'tanh',
                    'squeeze', 'power'])

    batch_size = mir_graph[0]['attrs']['shape'][0]
    inshape_H = mir_graph[0]['attrs']['shape'][2] if len(mir_graph[0]['attrs']['shape']) == 4 else -1
    inshape_W = mir_graph[0]['attrs']['shape'][3] if len(mir_graph[0]['attrs']['shape']) == 4 else -1
    if batch_size == 1 and (inshape_H < 1024 and inshape_W < 1024):
        VPU_Bandwidth_INT8 = 512
        VPU_Bandwidth_BF16 = 256
        PE_Array_MACs = 4096
        GLB_Bandwidth = 8192
        DDR_bandwidth = 16
        Core2Core_bandwidth = None
        W_GLB_Cache_bandwidth = 64
        W_GLB_Cache_memory = 2*8*1000000
    elif batch_size >= 4 or (batch_size == 1 and inshape_H >= 1024 and inshape_W >= 1024):
        VPU_Bandwidth_INT8 = 2048
        VPU_Bandwidth_BF16 = 1024
        PE_Array_MACs = 16384
        GLB_Bandwidth = 32768
        DDR_bandwidth = 64
        Core2Core_bandwidth = 512
        W_GLB_Cache_bandwidth = 256
        W_GLB_Cache_memory = 8*8*1000000
    else:
        raise RuntimeError('Not supported batch size {}'.format(batch_size))

    f_weight_INT8_cache_flag = False
    f_weight_BF16_cache_flag = False
    if weight_cache:
        weight_sum = 0
        for node in mir_graph[1:]:
            if ('conv2d' in node['op_type'] and node['op_type'] != 'nn.conv2d_transpose') or node['op_type'] == 'nn.dense':
                groups = node['attrs']['groups'] if 'groups' in node['attrs'] else 1
                weight_size = 1
                for item in node['attrs']['A_shape'][1]:
                    weight_size *= item
                if groups == 1 and sparsity:
                    in_c = node['attrs']['A_shape'][0][1]
                    if in_c <= 8:
                        weight_size = weight_size
                    elif in_c <= 16 and in_c > 8:
                        weight_size = weight_size/2
                    elif in_c > 16 and in_c <= 32:
                        weight_size = weight_size/4
                    elif in_c > 32 and in_c <= 64:
                        weight_size = weight_size/8
                    else:
                        weight_size = weight_size/16
                weight_sum += weight_size
            elif node['op_type'] == 'nn.conv2d_transpose':
                weight_size = 1
                for item in node['attrs']['A_shape'][1]:
                    weight_size *= item
                weight_sum += weight_size
        if weight_sum*8 <= W_GLB_Cache_memory:
            f_weight_INT8_cache_flag = True
        if weight_sum*16 <= W_GLB_Cache_memory:
            f_weight_BF16_cache_flag = True
        logging.info('Weight Cache Mode {}, Sparsity {}, Use cache INT8 {}'.format(weight_cache, sparsity, f_weight_INT8_cache_flag))
        logging.info('Weight Cache Mode {}, SParsity {}, Use cache BF16 {}'.format(weight_cache, sparsity, f_weight_BF16_cache_flag))
    Formula = namedtuple('Formula', ['VPU_Bandwidth_INT8', 'VPU_Bandwidth_BF16', 'PE_Array_MACs', 'GLB_Bandwidth',
                        'DDR_bandwidth', 'Core2Core_bandwidth', 'sparsity', 'W_GLB_Cache_bandwidth', 'f_weight_INT8_cache_flag',
                        'f_weight_BF16_cache_flag'])
    formula = Formula(VPU_Bandwidth_INT8, VPU_Bandwidth_BF16, PE_Array_MACs, GLB_Bandwidth, DDR_bandwidth, Core2Core_bandwidth, sparsity,
                    W_GLB_Cache_bandwidth, f_weight_INT8_cache_flag, f_weight_BF16_cache_flag)

    timing_INT8_list = []
    timing_BF16_list = []
    node_name_list = []
    conv2d_dict = {'f_op_INT8':[], 'f_I/O_INT8':[], 'f_DDR_INT8':[], 'f_max_others_INT8':[],
                   'f_op_BF16':[], 'f_I/O_BF16':[], 'f_DDR_BF16':[], 'f_max_others_BF16':[],
                   'f_Halo_INT8':[], 'f_Halo_BF16':[], 'total_FLOPs': 0, 'total_WNNZ': 0,
                   'PE_Array': [], 'VPU': [], 'transpose': []}
    for node in mir_graph:
        op_type = node['op_type']
        if op_type not in op_list:
            logging.warning('Node {} {} is not in supported op list.'.format(node['name'], op_type))
            continue
        if op_type.startswith('mir'):
            timing_INT8_mir = []
            timing_BF16_mir = []
            combined_op = op_type.split('.')[-1].split('_')
            for j in range(len(combined_op)):
                if combined_op[j] == 'conv2d':
                    op = 'nn.conv2d'
                elif combined_op[j] == 'scale':
                    op = 'multiply'
                elif combined_op[j] == 'bias':
                    if j == 1:
                        op = 'mir.bias_add'
                    elif j == 0:
                        op = 'nn.bias_add'
                elif combined_op[j] == 'relu':
                    op = 'nn.relu'
                else:
                    op = combined_op[j]
                timing_INT8, timing_BF16, conv2d_dict = calculate_node_timing(op, node, formula, conv2d_dict)
                timing_INT8_mir.append(timing_INT8)
                timing_BF16_mir.append(timing_BF16)
            logging.info('{} {}, INT8 {}, BF16{}'.format(node['name'], op_type, timing_INT8_mir, timing_BF16_mir))
            timing_INT8 = max(timing_INT8_mir)
            timing_BF16 = max(timing_BF16_mir)
            if 'conv2d' in op_type:
                if timing_INT8 != timing_INT8_mir[0]:
                    conv2d_dict['f_max_others_INT8'].append(node['name'])
                    if node['name'] in conv2d_dict['f_op_INT8']:
                        conv2d_dict['f_op_INT8'].remove(node['name'])
                    elif node['name'] in conv2d_dict['f_I/O_INT8']:
                        conv2d_dict['f_I/O_INT8'].remove(node['name'])
                    elif node['name'] in conv2d_dict['f_DDR_INT8']:
                        conv2d_dict['f_DDR_INT8'].remove(node['name'])
                    elif node['name'] in conv2d_dict['f_Halo_INT8']:
                        conv2d_dict['f_Halo_INT8'].remove(node['name'])
                if timing_BF16 != timing_BF16_mir[0]:
                    conv2d_dict['f_max_others_BF16'].append(node['name'])
                    if node['name'] in conv2d_dict['f_op_BF16']:
                        conv2d_dict['f_op_BF16'].remove(node['name'])
                    elif node['name'] in conv2d_dict['f_I/O_BF16']:
                        conv2d_dict['f_I/O_BF16'].remove(node['name'])
                    elif node['name'] in conv2d_dict['f_DDR_BF16']:
                        conv2d_dict['f_DDR_BF16'].remove(node['name'])
                    elif node['name'] in conv2d_dict['f_Halo_BF16']:
                        conv2d_dict['f_Halo_BF16'].remove(node['name'])
        else:
            timing_INT8, timing_BF16, conv2d_dict = calculate_node_timing(op_type, node, formula, conv2d_dict)
        node_name_list.append(node['name'])
        timing_INT8_list.append(timing_INT8)
        timing_BF16_list.append(timing_BF16)

        logging.info('Batch size {}. Theoretical timing INT8 of node {} {} is {} cycles'.format(batch_size, node['name'], op_type, timing_INT8))
        logging.info('Batch size {}. Theoretical timing BF16 of node {} {} is {} cycles'.format(batch_size, node['name'], op_type, timing_BF16))
    timing_INT8_sum = sum(timing_INT8_list)
    timing_BF16_sum = sum(timing_BF16_list)
    logging.info('Batch size {}. Theoretical timing INT8 sum is {} cycles'.format(batch_size, timing_INT8_sum))
    logging.info('Batch size {}. Theoretical timing BF16 sum is {} cycles'.format(batch_size, timing_BF16_sum))

    util_rate_INT8 = conv2d_dict['total_FLOPs']/(timing_INT8_sum * formula.PE_Array_MACs) * 100
    util_rate_BF16 = conv2d_dict['total_FLOPs']/(timing_BF16_sum * formula.PE_Array_MACs) * 100
    logging.info('Batch size {}. Util rate INT8 is {}'.format(batch_size, util_rate_INT8))
    logging.info('Batch size {}. Util rate BF16 is {}'.format(batch_size, util_rate_BF16))

    # PE_Array_timing = 1/(sum(conv2d_dict['PE_Array'])/800/1000000)*batch_size
    # VPU_timing = 1/(sum(conv2d_dict['VPU'])/800/1000000)*batch_size
    # transpose_timing = 1/(sum(conv2d_dict['transpose'])/800/1000000)*batch_size
    logging.info('PE_array:{}, VPU:{}, Transpose:{} cycles'.format(sum(conv2d_dict['PE_Array']), sum(conv2d_dict['VPU']), sum(conv2d_dict['transpose'])))

    INT8_percentage_list = []
    BF16_percentage_list = []
    for i in range(len(node_name_list)):
        INT8_percentage_list.append(timing_INT8_list[i]/timing_INT8_sum*100)
        BF16_percentage_list.append(timing_BF16_list[i]/timing_BF16_sum*100)
    x = range(len(node_name_list))
    x1_INT8 = [node_name_list.index(i) for i in conv2d_dict['f_op_INT8']]
    x1_INT8_height = [INT8_percentage_list[i] for i in x1_INT8]
    x2_INT8 = [node_name_list.index(i) for i in conv2d_dict['f_I/O_INT8']]
    x2_INT8_height = [INT8_percentage_list[i] for i in x2_INT8]
    x3_INT8 = [node_name_list.index(i) for i in conv2d_dict['f_DDR_INT8']]
    x3_INT8_height = [INT8_percentage_list[i] for i in x3_INT8]
    x4_INT8 = [node_name_list.index(i) for i in conv2d_dict['f_max_others_INT8']]
    x4_INT8_height = [INT8_percentage_list[i] for i in x4_INT8]
    x5_INT8 = [node_name_list.index(i) for i in conv2d_dict['f_Halo_INT8']]
    x5_INT8_height = [INT8_percentage_list[i] for i in x5_INT8]
    plt.figure(figsize=(16,9)  , dpi=200)
    rects_INT8 = plt.bar(x, height=INT8_percentage_list, width=0.8, color='gray', label='nn.softmax, transpose, add, multiply...')
    if x1_INT8:
        rects1_INT8 = plt.bar(x1_INT8, height=x1_INT8_height, width=0.8, color='red', label='maximum: nn.dense, nn.batch_matmul op time')
    if x2_INT8:
        rects2_INT8 = plt.bar(x2_INT8, height=x2_INT8_height, width=0.8, color='green', label='maximum: conv2d input I/O time')
    if x3_INT8:
        rects3_INT8 = plt.bar(x3_INT8, height=x3_INT8_height, width=0.8, color='lightskyblue', label='maximum: nn.dense weight I/O time')
    if x4_INT8:
        rects4_INT8 = plt.bar(x4_INT8, height=x4_INT8_height, width=0.8, color='gold', label='maximum: bias_add or relu')
    if x5_INT8:
        rects5_INT8 = plt.bar(x5_INT8, height=x5_INT8_height, width=0.8, color='purple', label='maximum: Halo')
    # plt.ylim(0,20)
    plt.ylabel('percentage %')
    plt.xticks([index for index in x], node_name_list, fontsize=5)
    plt.xlim(-1,len(node_name_list))
    # plt.xlabel("PE_Array: {}, VPU: {}, Transpose: {}, sentences per second.".format(PE_Array_timing, VPU_timing, transpose_timing))
    plt.title("The theoretical FPS of Batch {} Sparsity {} Cache {} INT8 is {}. Util rate {}".format(batch_size, sparsity, f_weight_INT8_cache_flag, 1/(timing_INT8_sum/800/1000000)*batch_size, util_rate_INT8))
    plt.legend(loc='upper right')
    INT8_result_name = os.path.join(output_path, "INT8_timing_result.png")
    plt.savefig(INT8_result_name)

    x1_BF16 = [node_name_list.index(i) for i in conv2d_dict['f_op_BF16']]
    x1_BF16_height = [BF16_percentage_list[i] for i in x1_BF16]
    x2_BF16 = [node_name_list.index(i) for i in conv2d_dict['f_I/O_BF16']]
    x2_BF16_height = [BF16_percentage_list[i] for i in x2_BF16]
    x3_BF16 = [node_name_list.index(i) for i in conv2d_dict['f_DDR_BF16']]
    x3_BF16_height = [BF16_percentage_list[i] for i in x3_BF16]
    x4_BF16 = [node_name_list.index(i) for i in conv2d_dict['f_max_others_BF16']]
    x4_BF16_height = [BF16_percentage_list[i] for i in x4_BF16]
    x5_BF16 = [node_name_list.index(i) for i in conv2d_dict['f_Halo_BF16']]
    x5_BF16_height = [BF16_percentage_list[i] for i in x5_BF16]
    plt.figure(figsize=(16,9)  , dpi=200)
    rects_BF16 = plt.bar(x, height=BF16_percentage_list, width=0.8, color='gray', label='max/avg_pool2d, mir.add_relu')
    if x1_BF16:
        rects1_BF16 = plt.bar(x1_BF16, height=x1_BF16_height, width=0.8, color='red', label='maximum: conv2d op time')
    if x2_BF16:
        rects2_BF16 = plt.bar(x2_BF16, height=x2_BF16_height, width=0.8, color='green', label='maximum: conv2d input I/O time')
    if x3_BF16:
        rects3_BF16 = plt.bar(x3_BF16, height=x3_BF16_height, width=0.8, color='lightskyblue', label='maximum: conv2d weight I/O time')
    if x4_BF16:
        rects4_BF16 = plt.bar(x4_BF16, height=x4_BF16_height, width=0.8, color='gold', label='maximum: bias_add or relu')
    if x5_BF16:
        rects5_BF16 = plt.bar(x5_BF16, height=x5_BF16_height, width=0.8, color='purple', label='maximum: Halo')
    # plt.ylim(0,20)
    plt.ylabel('percentage %')
    plt.xticks([index for index in x], node_name_list, fontsize=5)
    plt.xlim(-1,len(node_name_list))
    plt.xlabel("FLOPs {} W_NNZ {}".format(conv2d_dict['total_FLOPs'], conv2d_dict['total_WNNZ']))
    plt.title("The theoretical FPS of Batch {} Sparsity {} Cache {} BF16 is {}. Util rate {}".format(batch_size, sparsity, f_weight_BF16_cache_flag, 1/(timing_BF16_sum/800/1000000)*batch_size, util_rate_BF16))
    plt.legend(loc='upper right')
    BF16_result_name = os.path.join(output_path, "BF16_timing_result.png")
    plt.savefig(BF16_result_name)

    plt.show()

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Calculate theoretical time of Moffett model.',
                                         usage="calculate_timing.py IR_fused_for_CModel_graph IR_fused_for_CModel_params\n\
                                            i.e.: python3 calculate_flops.py IR_fused_for_CModel_graph.json")
    arg_parser.add_argument("IR_fused_for_CModel_graph", type=str, help="The path of IR fused graph for CModel.")
    arg_parser.add_argument("--log", type=str, default="theoretical_timing", help="The name of log file.")
    arg_parser.add_argument('--sparsity', type=bool, default=False, help="Whether compress the model. default: False.")
    arg_parser.add_argument('--weight_cache', type=bool, default=False, help="Whether use cache for weight. default: False.")
    arg_parser.add_argument('--output_path', type=str, default='theoretical_timing_result')
    args = arg_parser.parse_args()

    calculate_timing(args)
