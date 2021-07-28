#!/bin/bash
NUM_PROC=$1
shift
python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC train.py \
/home/haoc/datasets/RP2K \
--train-split train \
--val-split val \
--model swin_tiny_patch4_window7_224 \
--pretrained \
--num-classes 2388 \
--batch-size 64 \
--opt adam \
--weight-decay 0.0001 \
--sched cosine \
--lr 0.001 \
--min-lr 0.000001 \
--epochs 50 \
--warmup-epochs 3 \
--color-jitter 0.0 \
--drop-path 0.1 \
--amp \
--native-amp \
--sync-bn \
--smoothing 0.1 \
--output experiments/rp2k_swin_tiny_224_basicaug


