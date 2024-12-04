CHUNKS=$1

#---------------------------------------------------------------------------
###Open-MAGVIT2
# 1.5B Sampling
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=$IDX python3 sample.py \
        --ckpt "../upload_ckpts/Open-MAGVIT2/AR_256_XL/AR_256_XL.ckpt" \
        --o "./Open-MAGVIT2/XL_sample_gpu" \
        --config "configs/Open-MAGVIT2/gpu/imagenet_conditional_llama_XL.yaml" \
        -k 0,0 \
        -p 1.0,1.0 \
        -t 1.1,1.0 \
        -n 50 \
        --token_factorization \
        --batch_size 50 \
        --cfg_scale 1.75,1.85 \
        --global_seed 42 \
        --model Open-MAGVIT2 \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX &
done

wait

echo "combining"

### logdir format
python combine_npz.py --logdir "./Open-MAGVIT2/XL_sample_gpu/samples/top_k_0_0_temp_1.10_1.00_top_p_1.0_1.0_cfg_1.75_1.85/AR_256_XL.ckpt"

# #------------------------------------------------------------------------------------------------------------------
# #804M sampling
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=$IDX python3 sample.py \
        --ckpt "../upload_ckpts/Open-MAGVIT2/AR_256_L/AR_256_L.ckpt" \
        --o "./Open-MAGVIT2/L_sample_gpu" \
        --config "configs/Open-MAGVIT2/gpu/imagenet_conditional_llama_L.yaml" \
        -k 0,0 \
        -p 1.0,1.0 \
        -t 1.1,1.0 \
        -n 50 \
        --token_factorization \
        --batch_size 50 \
        --cfg_scale 1.9,1.9 \
        --global_seed 42 \
        --model Open-MAGVIT2 \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX &
done

wait

echo "combining"

# ### logdir format
python combine_npz.py --logdir "./Open-MAGVIT2/L_sample_gpu/samples/top_k_0_0_temp_1.10_1.00_top_p_1.0_1.0_cfg_1.9_1.9/AR_256_L.ckpt"

# #----------------------------------------------------------------------------------------------------------------------------
# # 337M sampling
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=$IDX python3 sample.py \
        --ckpt "../upload_ckpts/Open-MAGVIT2/AR_256_B/AR_256_B.ckpt" \
        --o "./Open-MAGVIT2/B_sample_gpu" \
        --config "configs/Open-MAGVIT2/gpu/imagenet_conditional_llama_B.yaml" \
        -k 0,0 \
        -p 1.0,1.0 \
        -t 1.0,1.0 \
        -n 50 \
        --token_factorization \
        --batch_size 50 \
        --cfg_scale 1.9,2.0 \
        --global_seed 42 \
        --model Open-MAGVIT2 \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX &
done

wait

echo "combining"

### logdir format
python combine_npz.py --logdir "./Open-MAGVIT2/B_sample_gpu/samples/top_k_0_0_temp_1.00_1.00_top_p_1.0_1.0_cfg_1.9_2.0/AR_256_B.ckpt"

# ###--------------------------------------------------------------------------------------------------------------------
# ### IBQ
# ### 342M
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=$IDX python3 sample.py \
        --ckpt "../upload_ckpts/IBQ/AR_256_B/AR_256_B.ckpt" \
        --o "./IBQ/B_sample_gpu" \
        --config "./configs/IBQ/gpu/imagenet_conditional_llama_B.yaml" \
        -k 0 \
        -p 1.0 \
        -n 50 \
        -t 1.10 \
        --batch_size 50 \
        --cfg_scale 2.25 \
        --model IBQ \
        --global_seed 42 \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX &
done

wait

echo "combining"

### logdir format
python combine_npz.py --logdir "./IBQ/B_sample_gpu/samples/top_k_0_temp_1.10_top_p_1.0_cfg_2.25/AR_256_B.ckpt"

# ### 649M
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=$IDX python3 sample.py \
        --ckpt "../upload_ckpts/IBQ/AR_256_L/AR_256_L.ckpt" \
        --o "./IBQ/L_sample_gpu" \
        --config "./configs/IBQ/gpu/imagenet_conditional_llama_L.yaml" \
        -k 0 \
        -p 1.0 \
        -n 50 \
        -t 1.00 \
        --batch_size 50 \
        --cfg_scale 2.0 \
        --global_seed 42 \
        --model IBQ \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX &
done

wait

echo "combining"

### logdir format
python combine_npz.py --logdir "./IBQ/L_sample_gpu/samples/top_k_0_temp_1.00_top_p_1.0_cfg_2.0/AR_256_L.ckpt"

# # ###
# # #1.1B
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=$IDX python3 sample.py \
        --ckpt "../upload_ckpts/IBQ/AR_256_XL/AR_256_XL.ckpt" \
        --o "./IBQ/XL_sample_gpu" \
        --config "./configs/IBQ/gpu/imagenet_conditional_llama_XL.yaml" \
        -k 0 \
        -p 1.0 \
        -n 50 \
        -t 1.15 \
        --batch_size 50 \
        --cfg_scale 2.35 \
        --model IBQ \
        --global_seed 42 \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX &
done

wait

echo "combining"

### logdir format
python combine_npz.py --logdir "./IBQ/XL_sample_gpu/samples/top_k_0_temp_1.15_top_p_1.0_cfg_2.35/AR_256_XL.ckpt"

# ### 
# ## 2.1B
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=$IDX python3 sample.py \
        --ckpt "../upload_ckpts/IBQ/AR_256_XXL/AR_256_XXL.ckpt" \
        --o "./IBQ/XXL_sample_gpu" \
        --config "./configs/IBQ/gpu/imagenet_conditional_llama_XXL.yaml" \
        -k 0 \
        -p 1.0 \
        -n 50 \
        -t 1.2 \
        --batch_size 50 \
        --cfg_scale 2.85 \
        --model IBQ \
        --global_seed 42 \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX &
done

wait

echo "combining"

### logdir format
python combine_npz.py --logdir "./IBQ/XXL_sample_gpu/samples/top_k_0_temp_1.20_top_p_1.0_cfg_2.85/AR_256_XXL.ckpt"