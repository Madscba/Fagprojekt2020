#!/bin/sh
#BSUB -J feature_script1
#BSUB -o feature_script1_%J.out
#BSUB -q gpuv100
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=128G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -u s173934@win.dtu.dk
#BSUB -N
# end of BSUB options

echo "Running script..."
source classifier-env/bin/activate
python3 New_CNN_HPC/CNN_final_new.py
echo "Done"