#/bin/bash

python evaluate.py \
        --model_name "basic_cnn" \
        --model "best_ckpts/dinos_with_top_best.pth" \
        --backbone "dinos" \
        --n_layers 2 \
        --topological_resolution 64