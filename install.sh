#!/bin/bash

# download model weights
mkdir weights
wget -O weights/checkpoint_95.pth https://www.dropbox.com/s/27gtxkmh2dlkho9/checkpoint_95.pth?dl=0

# create env and activate
conda create -n score-POCS python=3.8
conda activate score-POCS

# install dependencies
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt