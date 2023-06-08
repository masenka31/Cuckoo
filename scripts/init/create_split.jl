using DrWatson
using Cuckoo
using Flux
using JsonGrinder
using StatsBase
using DataFrames
using Random

# load data and labels
d = Dataset("cuckoo_small")
labels = d.family
const labelnames = sort(unique(labels))

##############################################################################################
################################ TIMESPLIT: HISTORY VS FUTURE ################################
##############################################################################################

using Dates
using Random

# extraction of indexes
b = d.date .< Date(2019,06,01)
history_ix = collect(1:length(d.family))[b]
future_ix = collect(1:length(d.family))[.!b]

n_seeds = 5

for s in 1:n_seeds
    # divide train/validation data from history
    Random.seed!(s)
    ixs = sample(history_ix, length(history_ix), replace=false)

    val_ix = ixs[1:length(ixs)÷3]
    train_ix = ixs[length(ixs)÷3+1:end]

    # sample extraction
    train_samples = d.samples[train_ix]
    val_samples = d.samples[val_ix]
    test_samples = d.samples[future_ix]

    df = DataFrame(
        :hash => vcat(train_samples, val_samples, test_samples),
        :split => vcat(
            repeat(["train"], length(train_samples)),
            repeat(["validation"], length(val_samples)),
            repeat(["test"], length(test_samples))
        )
    )

    safesave(datadir("timesplit/split_$s.csv"), df)
end

###################################################################################
################################ OTHER SPLITS HERE ################################
###################################################################################

# 60/20/20 split
for seed in 1:5
    ratios = (0.6,0.2,0.2)
    train_ix, val_ix, test_ix = train_val_test_ix(labels; ratios=ratios, seed=seed)

    train_samples = d.samples[train_ix]
    val_samples = d.samples[val_ix]
    test_samples = d.samples[test_ix]

    df = DataFrame(
        :hash => vcat(train_samples, val_samples, test_samples),
        :split => vcat(
            repeat(["train"], length(train_samples)),
            repeat(["validation"], length(val_samples)),
            repeat(["test"], length(test_samples))
        )
    )

    safesave(datadir("60-20-20/0$(seed)_split.csv"), df)
end

# 20/40/40 split
for seed in 1:5
    ratios = (0.2,0.4,0.4)
    train_ix, val_ix, test_ix = train_val_test_ix(labels; ratios=ratios, seed=seed)

    train_samples = d.samples[train_ix]
    val_samples = d.samples[val_ix]
    test_samples = d.samples[test_ix]

    df = DataFrame(
        :hash => vcat(train_samples, val_samples, test_samples),
        :split => vcat(
            repeat(["train"], length(train_samples)),
            repeat(["validation"], length(val_samples)),
            repeat(["test"], length(test_samples))
        )
    )

    safesave(datadir("20-40-40/0$(seed)_split.csv"), df)
end

# save labels as well with the hash
labels_df = DataFrame(
    :hash => d.samples,
    :family => d.family,
    :type => d.type
)
safesave(datadir("labels.csv"), labels_df)


### DateTime split
using Dates
# the split is created to be the end of the year 2018
# meaning that there are 41% training data

# extraction of indexes
b = d.date .< Date(2019,01,01)
train_ix = collect(1:length(d.family))[b]
else_ix = collect(1:length(d.family))[.!b]
using Random
Random.seed!(31)
ixs = sample(else_ix, length(else_ix), replace=false)
val_ix = ixs[1:length(ixs)÷2]
test_ix = ixs[length(ixs)÷2+1:end]

# sample extraction
train_samples = d.samples[train_ix]
val_samples = d.samples[val_ix]
test_samples = d.samples[test_ix]

df = DataFrame(
    :hash => vcat(train_samples, val_samples, test_samples),
    :split => vcat(
        repeat(["train"], length(train_samples)),
        repeat(["validation"], length(val_samples)),
        repeat(["test"], length(test_samples))
    )
)

safesave(datadir("time41/01_split.csv"), df)

# the next split is at the 57% mark dividing at 2019-04-01
b = d.date .< Date(2019,04,01)
train_ix = collect(1:length(d.family))[b]
else_ix = collect(1:length(d.family))[.!b]
using Random
Random.seed!(31)
ixs = sample(else_ix, length(else_ix), replace=false)
val_ix = ixs[1:length(ixs)÷2]
test_ix = ixs[length(ixs)÷2+1:end]

# sample extraction
train_samples = d.samples[train_ix]
val_samples = d.samples[val_ix]
test_samples = d.samples[test_ix]

df = DataFrame(
    :hash => vcat(train_samples, val_samples, test_samples),
    :split => vcat(
        repeat(["train"], length(train_samples)),
        repeat(["validation"], length(val_samples)),
        repeat(["test"], length(test_samples))
    )
)

safesave(datadir("time57/01_split.csv"), df)

####################################################################################
################################ Garcia data splits ################################
####################################################################################

d = Dataset("garcia")
labels = d.family
const labelnames = sort(unique(labels))

# samples are to be tested via cross-validation, probably
# I need to split those as such

cv_splits = map(x -> "/$x/", 0:9)
cv_indexes = map(x -> occursin.(x, d.samples .|> String), cv_splits)

for (i, class_indexes) in enumerate(cv_indexes)
    out_class = d.samples[class_indexes]
    in_classes = d.samples[.!class_indexes]

    n = length(in_classes)
    n_train = round(Int, n * 0.8)

    # ??? for seed in 1:5
    seed = i
    Random.seed!(seed)
    in_indexes = sample(1:n, n, replace=false)
    train_ix = in_indexes[1:n_train]
    val_ix = in_indexes[n_train+1:end]

    train_samples = in_classes[train_ix]
    val_samples = in_classes[val_ix]
    test_samples = out_class

    df = DataFrame(
        :hash => vcat(train_samples, val_samples, test_samples),
        :split => vcat(
            repeat(["train"], length(train_samples)),
            repeat(["validation"], length(val_samples)),
            repeat(["test"], length(test_samples))
        )
    )

    safesave(splitsdir("garcia/0$(seed)_split.csv"), df)
end