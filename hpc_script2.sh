#!/bin/sh
#BSUB -J test
#BSUB -o test_%J.out
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=32G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 10
# end of BSUB options

echo "Running script..."
module load python3
python hpc_test.py
