#!/bin/sh
#BSUB -J feature_script1
#BSUB -o feature_script1_%J.out
#BSUB -q hpc
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=14G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 10:00
#BSUB -u s173934@win.dtu.dk
#BSUB -N
# end of BSUB options

echo "Running script..."
source classifier-env/bin/activate
python3 CNN/trainCNN.py