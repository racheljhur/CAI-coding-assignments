# PACE-ICE SLURM Notes

## Batch training — Part B BiLSTM

Submit each fraction as its own job:

```bash
sbatch slurm/train_bilstm.sbatch 0.05
sbatch slurm/train_bilstm.sbatch 0.10
sbatch slurm/train_bilstm.sbatch 0.25
sbatch slurm/train_bilstm.sbatch 1.00
```
