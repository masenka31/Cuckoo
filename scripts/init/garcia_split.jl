# things
using Revise
using CSV, DataFrames
using Cuckoo
using Random
using StatsBase
using DrWatson

labels_df = CSV.read("/mnt/data/jsonlearning/datasets/garcia/meta.csv", DataFrame)
rename!(labels_df, :sha256 => :hash, :severity => :label)

for split_ix in 1:10
    seed = split_ix
    test = filter(:split => x -> x == split_ix, labels_df)
    train_val = filter(:split => x -> x != split_ix, labels_df)
    n = nrow(train_val)

    Random.seed!(seed)
    idx = sample(1:n, n, replace=false)

    val_ix = idx[1:length(idx)÷3]
    train_ix = idx[length(idx)÷3+1:end]
    
    train_h = train_val[train_ix, :hash]
    val_h = train_val[val_ix, :hash]
    test_h = test.hash

    df = DataFrame(
        :hash => vcat(train_h, val_h, test_h),
        :split => vcat(
            repeat(["train"], length(train_h)),
            repeat(["validation"], length(val_h)),
            repeat(["test"], length(test_h))
        )
    )
    safesave(splitsdir("garcia_split/split_$seed.csv"), df)
end

