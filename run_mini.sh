#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python  main.py \
--model_type vit_tiny \
--use_fp16 False \
--batch_size 32 \
--train_list /home/yoos/Documents/code/S3RT/S3RT/list/voxceleb1_train_list \
--val_list /data/voxceleb1/test_mini.txt \
--train_path /data/voxceleb \
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