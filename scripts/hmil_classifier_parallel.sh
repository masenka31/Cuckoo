MAX_SEED=$1
NUM_SAMPLES=$2

# Prompt the user
let "total_jobs = NUM_SAMPLES * MAX_SEED / 5"
echo "This will submit $total_jobs jobs. Do you want to proceed? (y/n)"
read answer

let "i = 0"
# submit to slurm
if [[ $answer == "y" || $answer == "Y" ]]; then
    echo "-------------------"
    for rep_start in $(seq 1 5 $NUM_SAMPLES)
    do
        echo "repetition start: $rep_start"
        let "rep_end = rep_start + 4"
        echo "repetition end:   $rep_end"
        # for seed in {1..$MAX_SEED}
        for seed in $(seq 1 1 $MAX_SEED)
        do
            echo "- seed: $seed"
            let "i = i + 1"
            sbatch ./hmil_classifier.sh $seed $rep_start $rep_end
        done
        echo "-------------------"
    done
else
    echo "Aborted submission."
fi