using DrWatson
include(srcdir("dataset.jl"))
include(srcdir("data.jl"))
include(srcdir("constructors.jl"))

using Flux
using Flux: @epochs

# load data and labels
d = Dataset()
labels = d.family
const labelnames = sort(unique(labels))

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