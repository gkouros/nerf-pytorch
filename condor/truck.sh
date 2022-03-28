#!/bin/bash

source /users/visics/gkouros/.bashrc
dir=/users/visics/gkouros/projects/nerf-repos/nerf-pytorch/
cd $dir

export PATH="/usr/local/cuda-11/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

python3 run_nerf.py --config configs/truck.txt

conda deactivate
