#!/bin/sh
#BSUB -J hpc_script2
#BSUB -o hpc_script2_%J.out
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=2G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 5:00
# end of BSUB options

echo "Running script..."
source classifier-env/bin/activate
python3 -m Classifier_experimentOne_isUsable/classifier_experiment_isUsable.py
