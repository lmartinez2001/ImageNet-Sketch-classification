#!/bin/bash

python main_swin.py \
    --finetune \
    --backbone "dinos" \
    --backbone_lr 0.0001 \
    --cls_lr 0.001 \
    --epochs 20 \
    --batch_size 128 \
    --model_ckpt_path "experiment/model_best.pth" \
    --experiment_name "[Finetune] New dinos class"