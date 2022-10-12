#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1  main.py \
--model_type vit_tiny \
--batch_size 64 \
--train_list list/voxceleb1_train_list \
--val_list list/trials.txt \
--train_path / \
--val_path /voxceleb1 \
--musan_path /musan_split \
--saveckp_freq 20 \
--imagenet_pretrain False \
--audioset_pretrain False \
--lr 1e-3 \
--min_lr 1e-5 \
--local_crops_number 4 \
--epochs 300 \
--warmup_epochs 20 > log.txt