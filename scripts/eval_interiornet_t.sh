#!/bin/bash
sh scripts/export_paths.sh

EXPNAME=interiornet_t_emm
POOL_SIZE=60
WEIGHTS=120000.pth

nice -n 19 python test_streetlearn_interiornet.py --exp ${EXPNAME} --disable_vis \
        --fusion_transformer --checkpoint_dir ${EXPNAME} --weights ${WEIGHTS} \
        --plot_curve --transformer_depth 6 \
        --datapath=$INTERIORNET_PATH --dataset interiornet --streetlearn_interiornet_type T 
        