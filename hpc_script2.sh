#!/bin/sh
#BSUB -J torch_gpu
#BSUB -o torch_gpu_%J.out
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 1
#BSUB -R "rusage[mem=16G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 60:00
# end of BSUB options

echo "Running script..."
module load python3/3.8.1
numpy/1.18.1-python-3.8.1-openblas-0.3.7
scipy/1.4.1-python-3.8.1

python3 hpc_test.py
