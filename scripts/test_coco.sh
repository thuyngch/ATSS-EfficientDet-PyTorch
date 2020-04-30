#!/usr/bin/env bash
set -e

CONFIG_FILE='configs/effdet/atss_effdet_d0.py'
WORK_DIR='work_dirs/atss_effdet_d0'
CHECKPOINT_FILE="${WORK_DIR}/atss_effdet_d0.pth"
RESULT_FILE="${WORK_DIR}/atss_effdet_d0.pkl"

GPUS=2
export CUDA_VISIBLE_DEVICES=2
PYTHON=${PYTHON:-"python"}

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
	--master_port=$((RANDOM + 10000)) \
	tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} \
	--launcher pytorch --out ${RESULT_FILE} --eval bbox
