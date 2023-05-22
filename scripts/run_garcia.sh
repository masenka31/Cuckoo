#!/bin/bash
# This runs Cockoo dataset classification.

# Run from this folder only.

LOG_DIR="${HOME}/logs/Cuckoo/garcia"
echo "$LOG_DIR"

if [ ! -d "$LOG_DIR" ]; then
	mkdir -p  $LOG_DIR
fi

# submit to slurm
for rep in $(seq 1 5 46)
do
    echo $rep
    echo $(($rep + 4))
    for seed in {1..5}
    do
        sbatch ./hmil_classifier.sh $seed $rep $(($rep + 4))
    done
done
