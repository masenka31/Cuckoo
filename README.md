# Cuckoo

Repository for experiments with the Cuckoo dataset.

## Basic info

### Data and train/validation/test splits

The data itself is saved at `/mnt/data/jsonlearning/Avast_cockoo`. There are two versions of the dataset: small and full.

The splits are saved to folder `/mnt/data/jsonlearning/splits` where they are separated based on the small or full version into
- `/mnt/data/jsonlearning/splits/Avast_cockoo`
- `/mnt/data/jsonlearning/splits/Avast_cuckoo_full`

There are other subfolders which distinguish the type of split
- `60-20-20` is the split of 60/20/20
- `20-40-40` is the split of 20/40/40
- `time41` is a split that cuts the train data to be all samples seen before 2019-01-01, resulting in 41% of samples being train samples
- `time57` is a split that cuts the data at 2019-04-01 resulting in 57% of samples being train,
validation and test splits for the time splits are randomly sampled from the rest.

### Feature files

If there are feature files that serve as an input to other models (such as classifier), they should be saved as `csv` files where the first column are the ids, thec column is called "hash" and all the other columns are features (do not have to have specific names).

### Results

The results for experiments are stored in folder `experiments`. The structure is as follows
```
experiments
|
|--- cuckoo_small
    |--- model_1_result_files
    |--- model_2_results_files
|
|--- cuckoo_full
    |--- model_1_result_files
    |--- model_2_results_files
```

## Label encoding

Labels can be encoded to numerical values. The key is simple: sort unique labels alphabetically and assign numbers from 1 to 10:
1. Adload
2. Emotet
3. HarHar
4. Lokibot
5. Qakbot
6. Swisyn
7. Trickbot
8. Ursnif
9. Zeus
10. njRAT

## Models

Currently implemented:
- Mill.jl multi-class classifier