#!/bin/bash
export MATTERPORT_PATH=/mnt/nvme1/data/matterport

# TRAINED
# CKPT=output/matterport_train/checkpoints/120000.pth

# PRETRAINED
CKPT=pretrained_models/matterport.pth
LAYERS=3
EXPNAME=matterport_debug_bs8

CUDA_VISIBLE_DEVICES=1 nice -n 19 python test_matterport.py --exp ${EXPNAME} --transformer_depth 6 \
        --fusion_transformer --ckpt $CKPT \
        --datapath=$MATTERPORT_PATH --layer_num $LAYERS
        
