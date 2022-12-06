#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition=cpu
#SBATCH --nodes=1 --ntasks-per-node=2 --cpus-per-task=1
#SBATCH --mem=12G

SEED=$1
REPSTART=$2
REPEND=$3

# module load Julia
module load Julia/1.7.3-linux-x86_64

# run from the scripts directory only!
julia --project ./hmil_dense_classifier.jl $SEED $REPSTART $REPEND
