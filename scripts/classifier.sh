#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --partition=cpufast
#SBATCH --nodes=1 --ntasks-per-node=2 --cpus-per-task=1
#SBATCH --mem=60G

MAX_SEED=$1

# module load Julia
module load Julia/1.7.3-linux-x86_64

julia --project --threads 10 ./classifier_seeds.jl ${MAX_SEED}