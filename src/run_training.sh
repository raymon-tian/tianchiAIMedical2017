#!/bin/bash
set -e

cd ./codebase

eps=100
python data_prepare.py --mode train
CUDA_VISIBLE_DEVICES=0 python train_main.py --model net_res18 -b 32 --epochs $eps --save-dir res18 
CUDA_VISIBLE_DEVICES=0 python train_main.py --model net_res18 -b 32 --resume results/res18/$eps.ckpt --test 1
cp ./results/res18/$eps.ckpt ../checkpoints/detector.ckpt

