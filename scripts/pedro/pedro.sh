#!/bin/bash
#SBATCH --partition=cpufast
#SBATCH --time=2:00:00
#SBATCH --mem=4G

module load Python/3.10.8-GCCcore-12.2.0
source .venv/bin/activate

START=$1
END=$2
python ./pedro_garcia.py $START $END