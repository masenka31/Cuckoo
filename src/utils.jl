using CSV

"""
    load_split(seed, ratio="timesplit")

Loads the correct split dataframe given the split seed and ratio.
"""
function load_split(seed, ratio = "timesplit")
    if ratio == "timesplit"
        split_df = CSV.read("/mnt/data/jsonlearning/splits/timesplit/split_$(seed).csv", DataFrame)
    elseif ratio == "garcia"
        split_df = CSV.read(splitsdir("garcia_split/split_$(seed).csv"), DataFrame)
    else
        error("""Ratio must be either
        - \"timesplit\" for timesplit on Brano Cuckoo or
        - \"garcia\" for one garcia split.""")
    end
    return split_df
end

"""
    read_labels_file(dataset::String)

Based on the provided dataset, loads the labels and prepares them such that there is a returned dataframe
with columns [:hash, :label].

Dataset can be
- cuckoo
- garcia
"""
function read_labels_file(dataset::String)
    if dataset == "cuckoo"
        labels_df = CSV.read("/mnt/data/jsonlearning/Avast_cuckoo/public_labels.csv", DataFrame)
        rename!(labels_df, :sha256 => :hash, :classification_family => :label)
        return labels_df[!, [:hash, :label]]
    elseif dataset == "garcia"
        labels_df = CSV.read("/mnt/data/jsonlearning/datasets/garcia/meta.csv", DataFrame)
        rename!(labels_df, :sha256 => :hash, :severity => :label)
        return labels_df[!, [:hash, :label]]
    end
end

"""
    load_split_features(dataset::String, feature_file::String; seed::Int=1, ratio=::String="timesplit")

Loads labels given dataset, features given the feature file (.csv), loads the split of the data.
Joins all dataframes together and returns train/validation/test splits.

The returned data is a tuple of train, validation, test data:
- data matrix
- labels
- hash
"""
function load_split_features(dataset::String, feature_file::String; seed::Int=1, ratio::String="timesplit")
    # read splits, labels, features
    split_df = load_split(seed, ratio)
    labels_df = read_labels_file(dataset)
    fdf = CSV.read(feature_file, DataFrame, header=false)

    rename!(fdf, 1 => :hash) # process features
    if fdf.hash[1][1] == '/'
        fdf.hash = map(x -> x[2:end-1], fdf.hash)
    end

    # join the dataframes
    df1 = innerjoin(fdf, split_df, on=:hash)
    df = innerjoin(df1, labels_df, on=:hash)

    # get train/validation/test features
    train = filter(:split => x -> x == "train", df)[:, 2:end]
    validation = filter(:split => x -> x == "validation", df)[:, 2:end]
    test = filter(:split => x -> x == "test", df)[:, 2:end]

    # and hashes
    hash = vcat(
        filter(:split => x -> x == "train", df).hash,
        filter(:split => x -> x == "validation", df).hash,
        filter(:split => x -> x == "test", df).hash
    )

    # returns a 3tuple of tuples: data matrix, labels, hash
    return (
        train[:, 1:end-3] |> Array |> transpose |> collect, train.label, filter(:split => x -> x == "train", df).hash,
        validation[:, 1:end-3] |> Array |> transpose |> collect, validation.label, filter(:split => x -> x == "validation", df).hash,
        test[:, 1:end-3] |> Array |> transpose |> collect, test.label, filter(:split => x -> x == "test", df).hash
    )
end

function load_indexes(d::Dataset; seed::Int=1, ratio="garcia")
    split_df = load_split(seed, ratio)

    # export the hash to map dataframes together
    hash = map(x -> x[end-68:end-5], d.samples)

    # prepare dataset for merge
    data_df = DataFrame(
        :i => 1:length(d.family),
        :hash => hash,
    )

    df = innerjoin(data_df, split_df, on=:hash)

    train_ix = filter(:split => x -> x == "train", df)[!, :i]
    val_ix = filter(:split => x -> x == "validation", df)[!, :i]
    test_ix = filter(:split => x -> x == "test", df)[!, :i]

    train_h = filter(:split => x -> x == "train", df).hash
    val_h = filter(:split => x -> x == "validation", df).hash
    test_h = filter(:split => x -> x == "test", df).hash

    return train_ix, val_ix, test_ix, train_h, val_h, test_h
end

function load_split_indexes(d::Dataset; seed::Int = 1, ratio = "timesplit")
    # load split
    split_df = load_split(seed, ratio)

    # export the hash to map dataframes together
    hash = map(x -> x[end-68:end-5], d.samples)
    
    # prepare dataset for merge
    data_df = DataFrame(
        :i => 1:length(d.family),
        :hash => hash,
    )

    df = innerjoin(data_df, split_df, on=:hash)

    train_ix = filter(:split => x -> x == "train", df)[!, :i]
    val_ix = filter(:split => x -> x == "validation", df)[!, :i]
    test_ix = filter(:split => x -> x == "test", df)[!, :i]

    if d.name == "garcia"
        Xtrain, ytrain, mtrain = d[train_ix]
        Xval, yval, mval = d[val_ix]
        Xtest, ytest, mtest = d[test_ix]

        return (
            Xtrain, ytrain, filter(:split => x -> x == "train", df).hash[mtrain],
            Xval, yval, filter(:split => x -> x == "validation", df).hash[mval],
            Xtest, ytest, filter(:split => x -> x == "test", df).hash[mtest]
        )
    else
        Xtrain, ytrain = d[train_ix]
        Xval, yval = d[val_ix]
        Xtest, ytest = d[test_ix]

        return (
            Xtrain, ytrain, filter(:split => x -> x == "train", df).hash,
            Xval, yval, filter(:split => x -> x == "validation", df).hash,
            Xtest, ytest, filter(:split => x -> x == "test", df).hash
        )
    end
end

### Normalization and standardization ###

function _minmax_transform(tr_x, val_x, test_x; dims=2, log_transform=true)
    if log_transform
        tr_x = log.(1 .+ tr_x)
        val_x = log.(1 .+ val_x)
        test_x = log.(1 .+ test_x)
    end
    
    mx = maximum(tr_x, dims=dims)
    mx[mx .== 0] .= 1
    tr_x_norm = tr_x ./ mx
    val_x_norm = val_x ./ mx
    test_x_norm = test_x ./ mx

    return tr_x_norm, val_x_norm, test_x_norm
end

function _standard_transform(tr_x, val_x, test_x; dims=2, log_transform=true)
    if log_transform
        tr_x = log.(1 .+ tr_x)
        val_x = log.(1 .+ val_x)
        test_x = log.(1 .+ test_x)
    end
    
    meanx = mean(tr_x, dims=dims)
    stdx = std(tr_x, dims=dims)
    stdx[stdx .== 0] .= 1
    tr_x_norm = (tr_x .- meanx) ./ stdx
    val_x_norm = (val_x .- meanx) ./ stdx
    test_x_norm = (test_x .- meanx) ./ stdx

    return tr_x_norm, val_x_norm, test_x_norm
end


function transform_data(tr_x, val_x, test_x; type="minmax", dims=2)
    if type == "minmax"
        return _minmax_transform(tr_x, val_x, test_x; dims=dims, log_transform=true)
    elseif type == "standard"
        return _standard_transform(tr_x, val_x, test_x; dims=dims, log_transform=true)
    elseif type == "log"
        return log.(1 .+ tr_x), log.(1 .+ val_x), log.(1 .+ test_x)
    else
        return tr_x, val_x, test_x
    end
end

### Hash to Int mapping tables
function cuckoo_hash_to_int(df)
    labels_df = CSV.read("/mnt/data/jsonlearning/Avast_cuckoo/public_labels.csv", DataFrame)
    labels_df.int_hash = UInt16.(collect(1:nrow(labels_df)))
    rename!(labels_df, :sha256 => :hash)
    labels_df = labels_df[!, Not([:classification_family, :classification_type, :date])]

    innerjoin(df, labels_df, on=:hash)[!, Not([:hash])]
end

function cuckoo_int_to_hash(df)
    labels_df = CSV.read("/mnt/data/jsonlearning/Avast_cuckoo/public_labels.csv", DataFrame)
    labels_df.int_hash = UInt16.(collect(1:nrow(labels_df)))
    rename!(labels_df, :sha256 => :hash)
    labels_df = labels_df[!, Not([:classification_family, :classification_type, :date])]

    innerjoin(df, labels_df, on=:int_hash)[!, Not([:int_hash])]
end