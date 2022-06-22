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
sh -c 'snakemake -j$SLURM_CPUS_PER_TASK --resources mem_mb=$SLURM_MEM_PER_NODE --use-conda --conda-frontend mamba $@' _ $@
