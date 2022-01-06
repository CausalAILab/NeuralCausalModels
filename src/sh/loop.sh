#!/bin/bash
#
#SBATCH --account=dsi
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_50
#SBATCH --mail-user=kl2792@columbia.edu
#SBATCH --output=out/logs/%j
#
# Loop the specified command.
#

module load cuda10.2/blas
module load cuda10.2/toolkit

logcmd()(echo $@; $@)

logcmd sbatch -d afterany:$SLURM_JOB_ID $0 $@
logcmd $@
