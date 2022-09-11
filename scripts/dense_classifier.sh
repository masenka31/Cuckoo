#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --partition=cpufast
#SBATCH --nodes=1 --ntasks-per-node=2 --cpus-per-task=1
#SBATCH --mem=40G

# provide the full path to feature file
FEATURES=$1

# module load Julia
module load Julia/1.7.3-linux-x86_64

# run from the scripts directory only!
julia --project ./dense_classifier.jl FEATURES