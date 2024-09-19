## GPU and NPU can use the same config for evaluation
python evaluation.py --config_file configs/gpu/imagenet_lfqgan_256_L.yaml --ckpt_path ../upload_ckpts/in1k_256_L/imagenet_256_L.ckpt --image_size 256

#python evaluation.py --config_file configs/npu/imagenet_lfqgan_256_L.yaml --ckpt_path ../upload_ckpts/in1k_256_L/imagenet_256_L.ckpt --image_size 256