#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29508}

export PYTHONPATH=`pwd`

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    train.py $CONFIG --launcher pytorch ${@:3}
