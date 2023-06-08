#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --partition=cpufast
#SBATCH --mem=4G

# provide the full path to feature file
SCRIPT=$1
MODEL=$2
FEATURES=$3
SEED=$4
REP=$5

# module load Julia
module load Julia/1.8.5-linux-x86_64

# run from the scripts directory only!
julia --project ./${SCRIPT}.jl $MODEL $FEATURES $SEED $REP
