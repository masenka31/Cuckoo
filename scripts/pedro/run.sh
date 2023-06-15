#!/bin/bash

for i in {0..19997..100}
do
    for j in $(seq $((i+100)) $((i+100)))
    do
        echo "i is $i and j is $j"
        sbatch pedro.sh $i $j
    done
done