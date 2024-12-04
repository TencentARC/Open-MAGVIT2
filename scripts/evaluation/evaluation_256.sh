## GPU and NPU can use the same config for evaluation
##Open-MAGVIT2
# python evaluation.py --config_file configs/Open-MAGVIT2/gpu/imagenet_lfqgan_256_L.yaml --ckpt_path ../upload_ckpts/Open-MAGVIT2/in1k_256_L/imagenet_256_L.ckpt --image_size 256 --model Open-MAGVIT2

##NPU
##Open-MAGVIT2
# python evaluation.py --config_file configs/Open-MAGVIT2/npu/imagenet_lfqgan_256_L.yaml --ckpt_path ../upload_ckpts/Open-MAGVIT2/in1k_256_L/imagenet_256_L.ckpt --image_size 256 --model Open-MAGVIT2

##IBQ NPU
## 16384
# python evaluation.py --config_file configs/IBQ/npu/imagenet_ibqgan_16384.yaml --ckpt_path ../upload_ckpts/IBQ/in1k_16384/imagenet256_16384.ckpt --image_size 256 --model IBQ

## 262144
# python evaluation.py --config_file configs/IBQ/npu/imagenet_ibqgan_262144.yaml --ckpt_path ../upload_ckpts/IBQ/in1k_262144/imagenet256_262144.ckpt --image_size 256 --model IBQ

## 8192
# python evaluation.py --config_file configs/IBQ/npu/imagenet_ibqgan_8192.yaml --ckpt_path ../upload_ckpts/IBQ/in1k_8192/imagenet256_8192.ckpt --image_size 256 --model IBQ

##1024
# python evaluation.py --config_file configs/IBQ/npu/imagenet_ibqgan_1024.yaml --ckpt_path ../upload_ckpts/IBQ/in1k_1024/imagenet256_1024.ckpt --image_size 256 --model IBQ

## IBQ GPU
## 16384
# python evaluation.py --config_file configs/IBQ/gpu/imagenet_ibqgan_16384.yaml --ckpt_path ../upload_ckpts/IBQ/in1k_16384/imagenet256_16384.ckpt --image_size 256 --model IBQ

## 262144
# python evaluation.py --config_file configs/IBQ/gpu/imagenet_ibqgan_262144.yaml --ckpt_path ../upload_ckpts/IBQ/in1k_262144/imagenet256_262144.ckpt --image_size 256 --model IBQ

## 8192
# python evaluation.py --config_file configs/IBQ/gpu/imagenet_ibqgan_8192.yaml --ckpt_path ../upload_ckpts/IBQ/in1k_8192/imagenet256_8192.ckpt --image_size 256 --model IBQ

## 1024
# python evaluation.py --config_file configs/IBQ/gpu/imagenet_ibqgan_1024.yaml --ckpt_path ../upload_ckpts/IBQ/in1k_1024/imagenet256_1024.ckpt --image_size 256 --model IBQ