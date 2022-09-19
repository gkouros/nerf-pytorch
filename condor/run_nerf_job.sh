#!/bin/bash
echo 'NeRF job started'
NAME=$1
EXP=$2

export PATH="/usr/local/cuda-11/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate nerf

dir=/users/visics/gkouros/projects/nerf-repos/nerf-pytorch/
cd $dir

python3 run_nerf.py --config configs/${NAME}.txt --expname "${NAME}/${EXP}"

conda deactivate
echo 'NeRF job terminated'
