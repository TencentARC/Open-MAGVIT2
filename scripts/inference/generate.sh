##Open-MAGVIT2
# python generate.py \
# --ckpt "../upload_ckpts/Open-MAGVIT2/AR_256_XL/AR_256_XL.ckpt" \
# -o "./visualize" \
# --config "configs/Open-MAGVIT2/npu/imagenet_conditional_llama_XL.yaml" \
# -k "0,0" \
# -p "0.96,0.96" \
# --token_factorization \
# -n 8 \
# -t "1.0,1.0" \
# --classes "207" \
# --batch_size 8 \
# --cfg_scale "4.0,4.0" \
# --model Open-MAGVIT2

##GPU
# python generate.py \
# --ckpt "../upload_ckpts/Open-MAGVIT2/AR_256_XL/AR_256_XL.ckpt" \
# -o "./visualize" \
# --config "configs/Open-MAGVIT2/gpu/imagenet_conditional_llama_XL.yaml" \
# -k "0,0" \
# -p "0.96,0.96" \
# --token_factorization \
# -n 8 \
# -t "1.0,1.0" \
# --classes "207" \
# --batch_size 8 \
# --cfg_scale "4.0,4.0" \
# --model Open-MAGVIT2

##IBQ
# python generate.py \
# --ckpt "../upload_ckpts/IBQ/AR_256_XXL/AR_256_XXL.ckpt" \
# -o "./visualize" \
# --config "configs/IBQ/npu/imagenet_conditional_llama_XXL.yaml" \
# -k 0 \
# -p 0.96 \
# -n 8 \
# -t 1.0 \
# --classes "207" \
# --batch_size 8 \
# --cfg_scale 4.0 \
# --model IBQ

# python generate.py \
# --ckpt "../upload_ckpts/IBQ/AR_256_XXL/AR_256_XXL.ckpt" \
# -o "./visualize" \
# --config "configs/IBQ/gpu/imagenet_conditional_llama_XXL.yaml" \
# -k 0 \
# -p 0.96 \
# -n 8 \
# -t 1.0 \
# --classes "207" \
# --batch_size 8 \
# --cfg_scale 4.0 \
# --model IBQ