#! /bin/bash
export MASTER_ADDR=${1:-localhost}
export MASTER_PORT=${2:-10055}
export NODE_RANK=${3:-0}
export OMP_NUM_THREADS=6

export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

echo $MASTER_ADDR
echo $MASTER_PORT

##NPU
NODE_RANK=$NODE_RANK python main.py fit --config configs/npu/imagenet_lfqgan_128_L.yaml

###GPU
# NODE_RANK=$NODE_RANK python main.py fit --config configs/gpu/imagenet_lfqgan_128_L.yaml