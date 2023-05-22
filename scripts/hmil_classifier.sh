#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition=cpu
#SBATCH --nodes=1 --ntasks-per-node=2 --cpus-per-task=1
#SBATCH --mem=20G

SEED=$1
REPSTART=$2
REPEND=$3

# module load Julia/1.7.3-linux-x86_64
module load Julia
echo $(julia --version)
export CUDA_VISIBLE_DEVICES='-1'

# run from the scripts directory only!
julia --project ./hmil_dense_classifier.jl $SEED $REPSTART $REPEND
# julia --project ./hmil_dense_classifier_garcia.jl $SEED $REPSTART $REPEND
