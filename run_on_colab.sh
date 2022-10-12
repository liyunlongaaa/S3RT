#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1  main.py \
--model_type vit_tiny \
--batch_size 32 \
--train_list list/voxceleb1_train_list \
--val_list list/trials.txt \
--train_path /home/yoos/Downloads \
--val_path /home/yoos/Downloads/voxceleb1 \
--musan_path /data/musan \
--saveckp_freq 5 \
--imagenet_pretrain False \
--audioset_pretrain False \
--lr 1e-3 \
--min_lr 1e-5 \
--output_dir ./output2 \
--local_crops_number 4 \
--epochs 100 \
--warmup_epochs 10 > log2.txt