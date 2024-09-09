python generate.py \
--ckpt "../upload_ckpts/AR_256_XL/AR_256_XL.ckpt" \
-o "./visualize" \
--config "configs/npu/imagenet_conditional_llama_XL.yaml" \
-k "0,0" \
-p "0.96,0.96" \
--token_factorization \
-n 8 \
-t "1.0,1.0" \
--classes "207" \
--batch_size 8 \
--cfg_scale "4.0,4.0" \

##GPU
# python generate.py \
# --ckpt "../upload_ckpts/AR_256_XL/AR_256_XL.ckpt" \
# -o "./visualize" \
# --config "configs/gpu/imagenet_conditional_llama_XL.yaml" \
# -k "0,0" \
# -p "0.96,0.96" \
# --token_factorization \
# -n 8 \
# -t "1.0,1.0" \
# --classes "207" \
# --batch_size 8 \
# --cfg_scale "4.0,4.0" \