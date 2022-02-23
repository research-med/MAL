#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29500}

export PYTHONPATH=`pwd`

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}
