#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition=cpu
#SBATCH --mem=40G

SEED=$1
REPSTART=$2
REPEND=$3

module load Julia/1.8.5-linux-x86_64
# module load Julia
# echo $(julia --version)
# export CUDA_VISIBLE_DEVICES='-1'

# # Define the output and error file paths
# OUTPUT_ERROR_FILE="/logs/hmil_$SLURM_JOB_ID}.txt"

# # Redirect both stdout and stderr to the output and error file
# exec > >(tee "$OUTPUT_ERROR_FILE") 2>&1

# run from the scripts directory only!
# julia --project ./hmil_dense_classifier.jl $SEED $REPSTART $REPEND
julia --project ./hmil_dense_classifier_garcia_softmax.jl $SEED $REPSTART $REPEND
