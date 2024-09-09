## GPU and NPU can use the same config for evaluation
# python evaluation.py --config_file configs/gpu/imagenet_lfqgan_128_L.yaml --ckpt_path ../upload_ckpts/in1k_128_L/imagenet_128_L.ckpt --image_size 128

python evaluation.py --config_file configs/npu/imagenet_lfqgan_128_L.yaml --ckpt_path ../upload_ckpts/in1k_128_L/imagenet_128_L.ckpt --image_size 128