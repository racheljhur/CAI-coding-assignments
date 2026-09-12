#!/bin/bash
#
# Ready to run.
# Submits a Part B BiLSTM training run on PACE-ICE using course-provided `train.py` CLI.
#
# Usage: sbatch slurm/train_bilstm.sbatch <train_fraction>
#   e.g. sbatch slurm/train_bilstm.sbatch 0.25

#SBATCH --job-name=a1a-bilstm
#SBATCH --gres=gpu:H100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

module load anaconda3
conda activate CAI_SO_FUN

FRACTION="${1:?Usage: sbatch slurm/train_bilstm.sbatch <fraction>}"

# Part A CRF doesn't need a GPU. Run it locally or on a login node instead.
# This job is for Part B BiLSTM.
python train.py \
    --model bilstm \
    --train-fraction "${FRACTION}" \
    --checkpoint-dir "checkpoints/bilstm_${FRACTION}" \
    --seed 42 \
    --tensorboard-logdir "runs/bilstm_${FRACTION}"
