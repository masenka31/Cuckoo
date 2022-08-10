#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --partition=cpufast
#SBATCH --nodes=1 --ntasks-per-node=2 --cpus-per-task=1
#SBATCH --mem=60G

# module load Julia
module load Julia/1.7.3-linux-x86_64

julia --project --threads 10 ./classifier.jl