#!/bin/sh
#BSUB -J torch_gpu
#BSUB -o torch_gpu_%J.out
#BSUB -q gpuk40
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 1
#BSUB -R "rusage[mem=16G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 5:00
# end of BSUB options

echo "Running script..."
source classifier-env/bin/activate
cd Fagprojekt2020/Classifier_experimentOne_isUsable
python3 classifier_experiment_isUsable.py
