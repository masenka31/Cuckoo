# Cuckoo

Repository for experiments with the Cuckoo dataset.

## Basic info

### Data and train/validation/test splits

The data itself is saved at `/mnt/data/jsonlearning/Avast_cockoo`. There are two versions of the dataset: small and full.

For experimentation, the dataset is split with 60%/20%/20% ratio for train/validation/test splits. Both split indexes and split IDs are provided in the folder `splits`:
- `split.csv` contains a DataFrame with IDs saved on columns `train, val, test` with the validation and test padded with "nothing"
- `split_indexes.csv` contains the same DataFrame with both IDs and indexes (columns `train, train_ix, val, val_ix, test, test_ix`)
- the same goes for individual `.csv` files for train/validation/test, if needed (these do not need to be padded)

There are also time based splits, two of them to be exact. The first cuts the train data to be all samples seen before 2019-01-01, resulting in 41% of samples being train samples. The second cuts the data at 2019-04-01 resulting in 57% of samples being train. Validation and test splits are randomly sampled from the rest.

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