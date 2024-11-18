#!/bin/bash

 python main_swin.py \
    --lr 0.01 \
    --warmup_epochs 3  \
    --epochs 50 \
    --batch_size 256 \
    --epochs_before_decay 7 \
    --weight_decay 0.01 \
    --backbone "dinol" \
    --experiment_name "Dinol dropout 0.5 and 2 fc 1152"
