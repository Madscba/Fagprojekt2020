#!/bin/sh
#BSUB -J full_dataset_script1
#BSUB -o full_dataset_script1_%J.out
#BSUB -q gpuv100
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=64G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 48:00
#BSUB -u s173934@win.dtu.dk
#BSUB -N
# end of BSUB options
