## NPU

##Open-MAGVIT2
# python reconstruct.py \
# --config_file "configs/Open-MAGVIT2/npu/imagenet_lfqgan_256_L.yaml" \
# --ckpt_path  ../upload_ckpts/Open-MAGVIT2/in1k_256_L/imagenet_256_L.ckpt \
# --save_dir "./visualize" \
# --version  "1k_Open_MAGVIT2" \
# --image_num 50 \
# --image_size 256 \
# --model Open-MAGVIT2 \

# ##IBQ
# python reconstruct.py \
# --config_file "configs/IBQ/npu/imagenet_ibqgan_262144.yaml" \
# --ckpt_path  ../upload_ckpts/IBQ/in1k_262144/imagenet256_262144.ckpt \
# --save_dir "./visualize" \
# --version  "1k_IBQ" \
# --image_num 50 \
# --image_size 256 \
# --model IBQ \

##GPU
# python reconstruct.py \
# --config_file "configs/Open-MAGVIT2/gpu/imagenet_lfqgan_256_L.yaml" \
# --ckpt_path  ../upload_ckpts/Open-MAGVIT2/in1k_256_L/imagenet_256_L.ckpt \
# --save_dir "./visualize" \
# --version  "1k_Open_MAGVIT2" \
# --image_num 50 \
# --image_size 256 \
# --model Open-MAGVIT2 \

# python reconstruct.py \
# --config_file "configs/IBQ/gpu/imagenet_ibqgan_262144.yaml" \
# --ckpt_path  ../upload_ckpts/IBQ/in1k_262144/imagenet256_262144.ckpt \
# --save_dir "./visualize" \
# --version  "1k_IBQ" \
# --image_num 50 \
# --image_size 256 \
# --model IBQ \