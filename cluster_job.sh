#!/usr/local_rwth/bin/zsh

cores = 4

#SBATCH --job-name=Brain_Connectivity_Pipeline
#SBATCH --ntasks=$cores
#SBATCH --mem-per-gpu=16G


#SBATCH --mail-type=ALL
#SBATCH --mail-user=damin.kuehn@rwth-aachen.de
#SBATCH --time=45:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate snakemake
snakemake -j$cores --use-conda all
