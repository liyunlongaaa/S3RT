#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python  main.py \
--model_type vit_tiny \
--use_fp16 False \
--lr 0.2 \
--min_lr 5e-5 \
--batch_size 200 \
--train_list list/voxceleb1_train_list \
--val_list list/trials.txt \
--train_path /home/yoos/Documents/data/voxceleb1 \
--val_path /home/yoos/Documents/data/voxceleb1 \
--musan_path /home/yoos/Documents/data/musan \
--saveckp_freq 1 \
--imagenet_pretrain False \
--audioset_pretrain False \
--local_crops_number 0 \
--epochs 100 \
--warmup_epochs 10 > log.txt