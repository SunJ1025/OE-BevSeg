#!/bin/bash

DATA_DIR="/home/oem/nuscenes"

EXP_NAME="rgb_segnet_test" # updated log dir

python train_nuscenes.py \
       --exp_name=${EXP_NAME} \
       --max_iters=2000 \
       --log_freq=40 \
       --dset='mini' \
       --batch_size=1 \
       --grad_acc=5 \
       --data_dir=$DATA_DIR \
       --log_dir='logs_nuscenes' \
       --ckpt_dir='checkpoints' \
       --res_scale=2 \
       --ncams=6 \
       --encoder_type='res101' \
       --do_rgbcompress=True \
       --device_ids=[0]


