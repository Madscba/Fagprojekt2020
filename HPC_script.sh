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

python3 Classifier_experimentOne_isUsable/hpc_test.py
