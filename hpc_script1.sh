#!/bin/sh
#BSUB -J hpc_script2
#BSUB -o hpc_script2_%J.out
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=64G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 48:00
# end of BSUB options

echo "Running script..."
source classifier-env/bin/activate
python3 Classifier_experimentOne_isUsable/Classifier_test.py
