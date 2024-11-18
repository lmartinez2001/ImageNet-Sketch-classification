#!/bin/bash

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python main_swin.py \
    --finetune \
    --backbone "dinos" \
    --backbone_lr 0.00001 \
    --cls_lr 0.0005 \
    --epochs 20 \
    --batch_size 128 \
    --model_ckpt_path "experiment/model_best.pth" \
    --experiment_name "[Finetune] New dinos class" \
    --device 1