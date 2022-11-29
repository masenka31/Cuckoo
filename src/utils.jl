"""
    load_split(seed, tr_ratio=60)

Loads the correct split dataframe given the split seed and ratio.

TODO: Add time split as well.
"""
function load_split(seed, tr_ratio=60)
    if tr_ratio == 60
        df = CSV.read(splitdir("60-20-20/0$(seed)_split.csv"), DataFrame)
    elseif tr_ratio == 20
        df = CSV.read(splitdir("20-40-40/0$(seed)_split.csv"), DataFrame)
    elseif tr_ratio == "time41"
        df = CSV.read(splitdir("time41/01_split.csv"), DataFrame)
    elseif tr_ratio == "time57"
        df = CSV.read(splitdir("time57/01_split.csv"), DataFrame)
    else
        error("Train ratio must be either 60, 20 for normal splits, or \"time41\", \"time57\" for time splits.")
    end
    return df
end

"""
    load_split_features(feature_file::String, labels_file::String, split_df::DataFrame=nothing; seed=1, tr_ratio=60)

Loads features from a csv file, splits them according to a given split,
returns train/validation/test splits with labels.
"""
function load_split_features(feature_file::String, labels_file::String; seed=1, tr_ratio=60)
    split_df = load_split(seed, tr_ratio)

    labels_df = CSV.read(labels_file, DataFrame)
    fdf = CSV.read(feature_file, DataFrame)
    rename!(fdf, 1 => :hash)
    if fdf.hash[1][1] == '/'
        fdf.hash = map(x -> x[2:end-1], fdf.hash)
    end

    df1 = innerjoin(fdf, split_df, on=:hash)
    df = innerjoin(df1, labels_df, on=:hash)

    train = filter(:split => x -> x == "train", df)[:, 2:end]
    validation = filter(:split => x -> x == "validation", df)[:, 2:end]
    test = filter(:split => x -> x == "test", df)[:, 2:end]

    hash = vcat(
        filter(:split => x -> x == "train", df).hash,
        filter(:split => x -> x == "validation", df).hash,
        filter(:split => x -> x == "test", df).hash
    )

    return (
        train[:, 1:end-3] |> Array |> transpose |> collect, train.family, filter(:split => x -> x == "train", df).hash,
        validation[:, 1:end-3] |> Array |> transpose |> collect, validation.family, filter(:split => x -> x == "validation", df).hash,
        test[:, 1:end-3] |> Array |> transpose |> collect, test.family, filter(:split => x -> x == "test", df).hash
    )
end

