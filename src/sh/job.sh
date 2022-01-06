#!/bin/bash
#
#SBATCH --account=dsi
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --output=out/logs/%j
#
# Run the specified command.
#

module load cuda10.2/blas
module load cuda10.2/toolkit

echo $@
$@
