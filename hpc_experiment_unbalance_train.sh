#!/bin/sh
#BSUB -J ex_unbalanced_train
#BSUB -o ex_unbalanced_train_%J.out
#BSUB -q gpuv100
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=64G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -u s173934@win.dtu.dk
#BSUB -N
# end of BSUB options

echo "Running script..."
source classifier-env/bin/activate
python3 Classifier_experimentOne_isUsable/experiment_unbalance_train.py
