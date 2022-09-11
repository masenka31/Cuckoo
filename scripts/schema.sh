#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --partition=cpulong
#SBATCH --nodes=1 --ntasks-per-node=2 --cpus-per-task=1
#SBATCH --mem=60G

# module load Julia
module load Julia/1.7.3-linux-x86_64

julia --project schema_extraction.jl