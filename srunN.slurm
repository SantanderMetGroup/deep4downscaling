#!/bin/bash
#SBATCH --job-name=notebooks
#SBATCH --partition=wngpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module load Miniconda3/24.7.1-0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /gpfs/users/garciafdez/yes/envs/bas_ent

cd notebooks


python -m nbconvert \
  --to notebook \
  --execute downscaling_vit.ipynb \
  --output downscaling_vit_EXECUTED.ipynb \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=bas_ent
