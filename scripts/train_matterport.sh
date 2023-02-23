#!/bin/bash
export MATTERPORT_PATH=/mnt/nvme1/data/matterport

LAYERS=2
BATCH=8
CROSSFEAT=1
EXPNAME=matterport_layer${LAYERS}_bs${BATCH}_unpretrain # _cf${CROSSFEAT}

CUDA_VISIBLE_DEVICES=1 nice -n 19 python train.py --name ${EXPNAME} --gpus=1 --batch=${BATCH} \
        --lr=5e-4 --fusion_transformer --transformer_depth 6 --cross_features \
        --w_tr 10 --w_rot 10 --steps 120000 --layer_num ${LAYERS} \
        --datapath=$MATTERPORT_PATH --dataset matterport 
# --no_ddp
# nice -n 19 python train.py --name ${EXPNAME} --gpus=10 --batch=6 \
#         --lr=5e-4 --fusion_transformer --transformer_depth 6 \
#         --w_tr 10 --w_rot 10 --steps 120000 \
#         --datapath=$MATTERPORT_PATH --dataset matterport 
