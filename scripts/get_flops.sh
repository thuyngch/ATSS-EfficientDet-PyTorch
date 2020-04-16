#!/usr/bin/env bash
set -e

HEIGHT=512
WIDTH=512
CONFIG_FILE='configs/effdet/atss_effdet_d0.py'

python tools/get_flops.py ${CONFIG_FILE} --shape ${WIDTH} ${HEIGHT}
