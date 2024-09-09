## NPU
python reconstruct.py \
--config_file "configs/npu/imagenet_lfqgan_256_L.yaml" \
--ckpt_path  ../upload_ckpts/in1k_256_L/imagenet_256_L.ckpt \
--save_dir "./visualize" \
--version  "1k" \
--image_num 50 \
--image_size 256 \


##GPU
# python reconstruct.py \
# --config_file "configs/gpu/imagenet_lfqgan_256_L.yaml" \
# --ckpt_path  ../upload_ckpts/in1k_256_L/imagenet_256_L.ckpt \
# --save_dir "./visualize" \
# --version  "1k" \
# --image_num 50 \
# --image_size 256 \