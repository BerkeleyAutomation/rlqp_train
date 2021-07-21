#!/bin/bash

# We don't include Eq QP below since it solves in 1 step causing
# qp_env to get stuck in an possibly infinite loop.

#--qp_env 'Random QP:10:2000' 'Eq QP:10:2000' 'Portfolio:5:150' 'Lasso:10:200' 'SVM:10:200' 'Huber:10:200' 'Control:10:100' \
         
python scripts/train.py \
       --num_epochs 25 \
       --debug \
       --qp_env 'Random QP:10:2000' 'Portfolio:5:150' 'Lasso:10:200' 'SVM:10:200' 'Control:10:100' \
       --qp_iters_per_step 200 \
       --max_ep_len 50 \
       --q_lr 0.0001 \
       --pi_lr 0.0001 \
       --replay_size 100000000 \
       --save_dir experiments/benchmarks_full_001

python scripts/convert.py \
       --save_dir experiments/benchmarks_full_001 \
       --traced_output benchmarks_full_001_traced.pt
