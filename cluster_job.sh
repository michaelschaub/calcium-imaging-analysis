#!/usr/local_rwth/bin/zsh

#SBATCH --job-name=Brain_Connectivity_Pipeline
#SBATCH --ntasks=4
#SBATCH --mem-per-gpu=16G
#SBATCH --mem=64G

#SBATCH --mail-type=ALL
#SBATCH --mail-user=damin.kuehn@rwth-aachen.de
#SBATCH --time=45:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate snakemake
snakemake -j4 --use-conda all
