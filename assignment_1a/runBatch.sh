#!/bin/bash

#SBATCH -J anyJob                       # job name
#SBATCH -A gts-skalidindi7-coda20
#SBATCH -N 1
#SBATCH --gres=gpu:V100:1
#SBATCH --ntasks-per-node=1             # specifies the number of cores per node
#SBATCH -t 00:10:00
#SBATCH -q inferno                      # queue name (where job is submitted)

# --gres=gpu:V100:1
# --cpus-per-task=10
# --mem-per-cpu=6G               # memory per core
# -p cpu-small

source ../inverse-psp-dynamics/.venv/bin/activate
#srun python procSveEnsembles.py
srun python train.py --model bilstm
