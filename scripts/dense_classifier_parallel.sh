#!/bin/bash
# This runs Cockoo dataset classification.

# Run from this folder only.
MODEL=$1 		# which model to run
FEATURES=$2     # feature file
NUM_SAMPLES=$3  # number of repetitions
NUM_CONC=$4		# number of concurrent tasks in the array job

LOG_DIR="${HOME}/logs/Cuckoo/features/${MODEL}"
echo "$LOG_DIR"

if [ ! -d "$LOG_DIR" ]; then
	mkdir -p  $LOG_DIR
fi

# submit to slurm
for rep in $(seq 1 1 $SEED)
do
    for seed in {1..5}
    do
        sbatch ./dense_classifier.sh $MODEL $FEATURES $seed $rep
    done
done