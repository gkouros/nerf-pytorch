#!/bin/bash


source /users/visics/gkouros/.bashrc
export PATH="/usr/local/cuda-11/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
conda activate nerf
dir=/users/visics/gkouros/projects/nerf-repos/nerf-pytorch/
cd $dir

SELF=$(readlink -f "${BASH_SOURCE[0]}")
BASENAME=$(basename "$SELF")
NAME=${BASENAME/%$'.sh'}

mkdir -p logs/$(NAME)_test

python3 run_nerf.py --config configs/$(NAME).txt

conda deactivate
