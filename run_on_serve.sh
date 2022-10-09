#!/bin/bash
'''
多机多卡
python -m torch.distributed.launch --nproc_per_node=每台机子gpu数
           --nnodes=2 --node_rank=0(不同机子要修改) --master_addr="192.168.1.1"(主节点的ip，也不能随便写)
           --master_port=1234 main.py (--arg1 --arg2 --arg3
           and all other arguments of your training script)
'''
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3  main.py \
--model_type vit_tiny \
--batch_size 64 \
--train_list /home/cxwang/code/S3RT/list/train_vox2_list.txt \
--val_list /export/corpus/Voxceleb/Voxceleb_tar/VoxCeleb1/voxceleb1_test.txt \
--train_path /export/corpus/Voxceleb/Voxceleb_tar/VoxCeleb2/dev/aac/ \
--val_path /export/corpus/Voxceleb/Voxceleb_tar/VoxCeleb1/voxceleb1_wav/ \
--musan_path /export/corpus/musan/musan/ \
--saveckp_freq 2 \
--imagenet_pretrain False \
--audioset_pretrain False \
--local_crops_number 4 \
--epochs 50 \
--warmup_epochs 5 > log.txt