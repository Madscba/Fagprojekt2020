#!/bin/sh
#BSUB -J torch_gpu
#BSUB -o torch_gpu_%J.out
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 1
#BSUB -R "rusage[mem=10G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 1
# end of BSUB options

echo "Running script..."
module load python3
python3 hpc_test.py
