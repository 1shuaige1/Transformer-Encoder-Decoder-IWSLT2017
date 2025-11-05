#!/bin/bash

python src/train.py \
  --epochs 10 \
  --batch_size 64 \
  --d_model 256 \
  --n_heads 4 \
  --n_layers 2 \
  --d_ff 1024 \
  --dropout 0.1 \
  --lr 3e-4 \
  --max_len 128 \
  --seed 42 \
  --limit_train_samples 48880 \
  --device cuda \
  --save_dir results
