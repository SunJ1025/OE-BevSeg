
#!/bin/bash

DATA_DIR="/home/oem/nuscenes"
# there should be ${DATA_DIR}/full_v1.0/
# and also ${DATA_DIR}/mini

MODEL_NAME="vov_mamba_bi_center_deform"  # rgb12

EXP_NAME="24000" # evaluate rgb00 model

python eval_nuscenes.py \
       --batch_size=1 \
       --exp_name=${EXP_NAME} \
       --dset='trainval' \
       --data_dir=$DATA_DIR \
       --log_dir='logs_eval_nuscenes' \
       --init_dir="checkpoints/${MODEL_NAME}" \
       --res_scale=2 \
       --step=$EXP_NAME \
       --encoder_type='vov99' \
       --device_ids=[0] \
#       --use_radar=True \
#       --use_metaradar=True