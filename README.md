# Cuckoo

Repository for experiments with the Cuckoo dataset.

## Basic info

### Data and train/validation/test splits

The data is saved at `/mnt/data/jsonlearning/`. There are three datasets in the works, currently:
- Cuckoo small (`Avast_cuckoo`)
- Cuckoo full (`Avast_cuckoo_full`)
- Garcia (`garcia`)

The timesplits are currently saved in folder
```/mnt/data/jsonlearning/splits```

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

**Encoding**

The saved files have certain encodings to make the saved CSVs smaller. First is the labels encoding, where labels are encoded into numerical values. The key is simple: sort unique labels alphabetically and assign numbers from 1 to 10 (for Cuckoo):
- 1: Adload
- 3: HarHar
- 4: Lokibot
- 2: Emotet
- 5: Qakbot
- 6: Swisyn
- 7: Trickbot
- 8: Ursnif
- 9: Zeus
- 10: njRAT

Also, in the CSV files, the splits are saved directly (train/validation/test), so that evaluation is faster. Those are numerically encoded to Int8 values as well with a simple code
- 0: train
- 1: validation
- 2: test

## Models

Currently implemented:

- feature model = HMIL
    - classification model:
        - neural network classifier
- feature model = Pedro's features
    - classification model:
        - neural network classifier
        - XGBoost
- feature model = Vasek's features
    - classification model:
        - neural network classifier
        - XGBoost