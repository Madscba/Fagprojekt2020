#!/bin/sh
#BSUB -J balance
#BSUB -o balance%J.out
#BSUB -q gpuv100
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=30G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 12:00
#BSUB -u s173934@win.dtu.dk
#BSUB -N
# end of BSUB options

echo "Running script..."
source classifier-env/bin/activate
python3 New_CNN_HPC/get_balance.py
echo "Done"