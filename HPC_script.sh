#!/bin/sh
#BSUB -J torch_gpu
#BSUB -o torch_gpu_%J.out
#BSUB -q hpc
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 1
#BSUB -R "rusage[mem=16G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 60:00
# end of BSUB options

echo "Running script..."
cd ../../../../work3/s173934/Fagprojekt
source classifier-env/bin/activate
python3 Fagprojekt2020/Classifier_experimentOne_isUsable/hpc_test.py
