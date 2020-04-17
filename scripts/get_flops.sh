#!/usr/bin/env bash
set -e

HEIGHT=640
WIDTH=640
CONFIG_FILE='configs/effdet/atss_effdet_d1.py'

CUDA_VISIBLE_DEVICES=2 python tools/get_flops.py ${CONFIG_FILE} --shape ${WIDTH} ${HEIGHT}
