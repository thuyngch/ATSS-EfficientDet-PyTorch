#!/usr/bin/env bash
set -e

SEED=0
GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
CONFIG_FILE='configs/effdet/atss_effdet_d0.py'

PYTHON=${PYTHON:-"python"}
${PYTHON} -m torch.distributed.launch --nproc_per_node=$GPUS \
	--master_port=$((RANDOM + 10000)) \
	tools/train.py ${CONFIG_FILE} \
	--launcher pytorch --seed $SEED --validate
