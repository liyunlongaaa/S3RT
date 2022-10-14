#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1  main.py \
--model_type vit_tiny \
--batch_size 64 \
--train_list /content/S3RT/list/voxceleb1_train_list \
--val_list /content/S3RT/list/trials.txt \
--train_path /content \
--val_path /content/voxceleb1 \
--musan_path /content/musan_split \
--saveckp_freq 10 \
--imagenet_pretrain False \
--audioset_pretrain False \
--lr 1e-4 \
--min_lr 1e-6 \
--output_dir /content/drive/MyDrive/exp2 \
--local_crops_number 4 \
--epochs 500 \
--warmup_epochs 10 > log2.txt
