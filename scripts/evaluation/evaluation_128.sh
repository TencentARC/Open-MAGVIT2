## GPU and NPU can use the same config for evaluation
# python evaluation.py --config_file configs/Open-MAGVIT2/gpu/imagenet_lfqgan_128_L.yaml --ckpt_path ../upload_ckpts/Open-MAGVIT2/in1k_128_L/imagenet_128_L.ckpt --image_size 128 --model Open-MAGVIT2

##NPU
##Open-MAGVIT2
python evaluation.py --config_file configs/Open-MAGVIT2/npu/imagenet_lfqgan_128_L.yaml --ckpt_path ../upload_ckpts/Open-MAGVIT2/in1k_128_L/imagenet_128_L.ckpt --image_size 128 --model Open-MAGVIT2