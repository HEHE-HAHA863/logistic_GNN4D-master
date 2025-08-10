#!/bin/sh
#BSUB -n 14
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -J random
#BSUB -m gpu11
nvidia-smi > gpu_info.out
python ./logistic_reg.py