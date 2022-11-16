#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python  main.py \
--eval \
--model_type vit_tiny \
--use_fp16 False \
--lr 1e-5 \
--min_lr 1e-6 \
--batch_size 8 \
--train_list list/voxceleb1_train_list \
--val_list list/trials.txt \
--train_path /data/voxceleb \
--val_path /data/voxceleb/voxceleb1 \
--musan_path /data/musan \
--saveckp_freq 1 \
--imagenet_pretrain False \
--audioset_pretrain False \
--local_crops_number 0 \
--epochs 100 \
--warmup_epochs 10 > log.txt