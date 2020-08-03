#docker run -ti -v /home:/home xhxian/frontendcompiler:0.0.1 /bin/bash

#python3 examples/test_complier.py mxnet models/resnet50v1b/resnet50v1b --epoch 0

#python3 examples/test_model_complexity.py \
#		--graph moffett_ir/IR_for_reconstruct_graph.json \
#		--params moffett_ir/IR_for_reconstruct_params.npz

#python3 examples/test_calibration.py -graph moffett_ir/IR_fused_for_CModel_graph.json -param moffett_ir/IR_fused_for_CModel_params.npz -input models/images/calibration -ppc configs/mxnet_imagenet_trans.json -o calibrations/resnet50_v1b.json --use-kl

#python3 examples/test_quantize_error.py --config-file configs/resnet50_v1b.yml

#python3 examples/test_pytorch_reconstructor.py --graph moffett_ir/IR_for_reconstruct_graph.json --params moffett_ir/IR_for_reconstruct_params.npz

#python3 examples/test_tensorflow_reconstructor.py --graph moffett_ir/IR_for_reconstruct_graph.json --params moffett_ir/IR_for_reconstruct_params.npz --save_path examples/tf_reconstruct.pb

python3 examples/test_imagenet_quantize_acc.py --config-file configs/resnet50_v1b.yml

