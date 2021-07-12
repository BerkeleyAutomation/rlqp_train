#!/bin/bash

python scripts/train.py \
       --num_epochs 25 \
       --debug \
       --qp_env 'Random QP:10:2000' 'Eq QP:10:2000' 'Portfolio:5:150' 'Lasso:10:200' 'SVM:10:200' 'Huber:10:200' 'Control:10:100' \
       --max_ep_len 50 \
       --save_dir experiments/benchmarks_full_001
