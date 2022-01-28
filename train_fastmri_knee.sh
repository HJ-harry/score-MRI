#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

python main_fastmri.py \
 --config=configs/ve/fastmri_knee_320_ncsnpp_continuous_multi.py \
 --eval_folder=eval/fastmri_multicoil_knee_320 \
 --mode='train'  \
 --workdir=workdir/fastmri_multicoil_knee_320