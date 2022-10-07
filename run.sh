#!/bin/bash
'''
多机多卡
python -m torch.distributed.launch --nproc_per_node=每台机子gpu数
           --nnodes=2 --node_rank=0(不同机子要修改) --master_addr="192.168.1.1"(主节点的ip，也不能随便写)
           --master_port=1234 main.py (--arg1 --arg2 --arg3
           and all other arguments of your training script)
'''
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1  main.py \
--batch_size 2 \
--train_list /data/voxceleb2/train_mini.txt \
--val_list /data/voxceleb1/test_mini.txt \
--train_path /data/voxceleb2 \
--val_path /data/voxceleb1/VoxCeleb1/voxceleb1_wav \
--musan_path /data/musan \
--saveckp_freq 2 \
--imagenet_pretrain False \
--audioset_pretrain False \
--local_crops_number 4 \
--epochs 4 \
--warmup_epochs 2