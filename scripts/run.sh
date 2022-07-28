#!/bin/bash
# This runs Cockoo dataset classification.

# Run from this folder only.
MODEL=$1 		# which model to run
NUM_SAMPLES=$2  # number of repetitions
MAX_SEED=$3		# how many folds over dataset
NUM_CONC=$4		# number of concurrent tasks in the array job

LOG_DIR="${HOME}/logs/Cockoo/${MODEL}"
echo "$LOG_DIR"

if [ ! -d "$LOG_DIR" ]; then
	mkdir -p  $LOG_DIR
fi

# submit to slurm
sbatch \
--array=1-${NUM_SAMPLES}%${NUM_CONC} \
--output="${LOG_DIR}/%A_%a.out" \
    ./${MODEL}.sh $MAX_SEED