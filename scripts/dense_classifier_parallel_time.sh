#!/bin/bash
# This runs Cockoo dataset classification.

# Run from this folder only.
SCRIPT=$1       # which julia script to run
MODEL=$2                # name of the model
FEATURES=$3     # path to feature file
NUM_SAMPLES=$4  # number of repetitions
RATIO=$5

LOG_DIR="${HOME}/logs/Cuckoo/features/${SCRIPT}/${MODEL}"
echo "$LOG_DIR"

if [ ! -d "$LOG_DIR" ]; then
        mkdir -p  $LOG_DIR
fi

# submit to slurm
for rep in $(seq 1 1 $NUM_SAMPLES)
do
    sbatch ./dense_classifier.sh $SCRIPT $MODEL $FEATURES 1 $rep $RATIO
done
