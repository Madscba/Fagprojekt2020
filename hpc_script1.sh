#!/bin/sh
#BSUB -J classifier_results
#BSUB -o classifier_results_%J.out
#BSUB -q hpc
#BSUB -n 6
#BSUB -R "rusage[mem=70G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 48:00
#BSUB -u s173934@win.dtu.dk
#BSUB -N 
# end of BSUB options

echo "Running script..."
source classifier-env/bin/activate
python3 Classifier_experimentOne_isUsable/Classifier_test.py
