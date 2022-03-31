#!/bin/bash
echo 'NeRF job started'
NAME=$1
source /users/visics/gkouros/.bashrc
export PATH="/usr/local/cuda-11/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
conda activate nerf
dir=/users/visics/gkouros/projects/nerf-repos/nerf-pytorch/
cd $dir

python3 run_nerf.py --config configs/${NAME}.txt

conda deactivate
echo 'NeRF job terminated'
