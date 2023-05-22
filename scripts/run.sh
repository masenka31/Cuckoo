#!/bin/bash
# This runs Cockoo dataset classification.

# Run from this folder only.
MODEL=$1 		# which model to run
NUM_SAMPLES=$2  # number of repetitions
NUM_CONC=$3		# number of concurrent tasks in the array job

LOG_DIR="${HOME}/logs/Cuckoo/new/${MODEL}"
echo "$LOG_DIR"

if [ ! -d "$LOG_DIR" ]; then
	mkdir -p  $LOG_DIR
fi

# submit to slurm
sbatch \
--array=1-${NUM_SAMPLES}%${NUM_CONC} \
--output="${LOG_DIR}/%A_%a.out" \
    ./${MODEL}.sh