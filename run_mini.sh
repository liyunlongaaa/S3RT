#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python  main.py \
--model_type vit_tiny \
--use_fp16 False \
--batch_size 16 \
--train_list /data/voxceleb2/train_mini.txt \
--val_list /data/voxceleb1/test_mini.txt \
--train_path /data/voxceleb/voxceleb2 \
--val_path /data/voxceleb1/VoxCeleb1/voxceleb1_wav \
--musan_path /data/musan \
--saveckp_freq 2 \
--imagenet_pretrain False \
--audioset_pretrain False \
--local_crops_number 4 \
--num_workers 8 \
--epochs 10 \
--output_dir ./outputt \
--warmup_epochs 1 > log4.txt