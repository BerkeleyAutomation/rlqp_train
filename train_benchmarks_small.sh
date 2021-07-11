#!/bin/bash

python scripts/train.py \
       --num_epochs 25 \
       --qp_env 'Random QP:10:100' 'Portfolio:5:15' 'Lasso:10:20' 'SVM:10:20' 'Control:10:10' \
       --save_dir experiments/benchmarks_small_001
