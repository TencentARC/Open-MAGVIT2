CHUNKS=$1

#---------------------------------------------------------------------------
# 1.5B Sampling
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=$IDX python3 sample.py \
        --ckpt "../upload_ckpts/AR_256_XL/AR_256_XL.ckpt" \
        --o "./XL_sample_gpu" \
        --config "configs/gpu/imagenet_conditional_llama_XL.yaml" \
        -k 0,0 \
        -p 1.0,1.0 \
        -t 1.1,1.0 \
        -n 50 \
        --token_factorization \
        --batch_size 50 \
        --cfg_scale 1.75,1.85 \
        --global_seed 42 \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX &
done

wait

echo "combining"

### logdir format
python combine_npz.py --logdir "./XL_sample_gpu/samples/top_k_0_0_temp_1.10_1.00_top_p_1.0_1.0_cfg_1.75_1.85/AR_256_XL.ckpt"

#------------------------------------------------------------------------------------------------------------------
#804M sampling
# for IDX in $(seq 0 $((CHUNKS-1))); do
#     CUDA_VISIBLE_DEVICES=$IDX python3 sample.py \
#         --ckpt "../upload_ckpts/AR_256_L/AR_256_L.ckpt" \
#         --o "./L_sample_gpu" \
#         --config "configs/gpu/imagenet_conditional_llama_L.yaml" \
#         -k 0,0 \
#         -p 1.0,1.0 \
#         -t 1.1,1.0 \
#         -n 50 \
#         --token_factorization \
#         --batch_size 50 \
#         --cfg_scale 1.9,1.9 \
#         --global_seed 42 \
#         --num_chunks $CHUNKS \
#         --chunk_idx $IDX &
# done

# wait

# echo "combining"

# ### logdir format
# python combine_npz.py --logdir "./L_sample_gpu/samples/top_k_0_0_temp_1.10_1.00_top_p_1.0_1.0_cfg_1.9_1.9/AR_256_L.ckpt"

#----------------------------------------------------------------------------------------------------------------------------
#337M sampling
# for IDX in $(seq 0 $((CHUNKS-1))); do
#     CUDA_VISIBLE_DEVICES=$IDX python3 sample.py \
#         --ckpt "../upload_ckpts/AR_256_B/AR_256_B.ckpt" \
#         --o "./B_sample_gpu" \
#         --config "configs/gpu/imagenet_conditional_llama_B.yaml" \
#         -k 0,0 \
#         -p 1.0,1.0 \
#         -t 1.0,1.0 \
#         -n 50 \
#         --token_factorization \
#         --batch_size 50 \
#         --cfg_scale 1.9,2.0 \
#         --global_seed 42 \
#         --num_chunks $CHUNKS \
#         --chunk_idx $IDX &
# done

# wait

# echo "combining"

# ### logdir format
# python combine_npz.py --logdir "./B_sample_gpu/samples/top_k_0_0_temp_1.00_1.00_top_p_1.0_1.0_cfg_1.9_2.0/AR_256_B.ckpt"
