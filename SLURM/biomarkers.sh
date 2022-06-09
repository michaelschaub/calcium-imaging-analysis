#!/usr/local_rwth/bin/zsh

#SBATCH --job-name=Brain_Connectivity_Pipeline
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

#SBATCH --mail-type=ALL
#SBATCH --mail-user=mail@example.de
#SBATCH --time=10:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate snakemake
snakemake -j8 --use-conda --rerun-incomplete best_features
