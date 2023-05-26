# Cuckoo

Repository for experiments with the Cuckoo dataset.

## Basic info

### Data and train/validation/test splits

The data itself is saved at `/mnt/data/jsonlearning/Avast_cuckoo`. There are two versions of the dataset: small and full.

The timesplits are currently saved in folder `/home/maskomic/projects/Cuckoo/data/timesplit`. They will be moved to a general folder once the repository is moved there as well.

### Feature files

If there are feature files that serve as an input to other models (such as classifier), they should be saved as `csv` files where the first column are the ids, the ccolumn is called "hash" and all the other columns are features (do not have to have specific names).

*Note: No label column is needed! The labels are mapped from the hash ids.*

### Results

The results for experiments are stored in folder `experiments/results`. The structure is as follows
```
results
|
|--- feature_extractor_1
    |--- model_1
        |--- result_file_1
        |--- result_file_2
        ...
    |--- model_2
        |--- result_file_1
        |--- result_file_2
        ...
|
|--- feature_extractor_2
    |--- model_1
        |--- result_file_1
        |--- result_file_2
        ...
    |--- model_2
        |--- result_file_1
        |--- result_file_2
        ...
```

Name of the feature extractor is used to create a general folder. The next folder contains results files for a particular classification model. The results are saved in two files: `bson` file and `csv` file:
- `bson` file contains metadata: parameters of features, model, seeds etc.
- `csv` file contains results: hash, ground truth, predicted labels, split names, and softmax output.

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
- standard classifier on Pedro's features
- standard classifier on Vasek's features