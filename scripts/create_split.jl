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

ratios = (0.6,0.2,0.2)
seed = 31
train_ix, val_ix, test_ix = train_val_test_ix(labels; ratios=ratios, seed=seed)

train_samples = d.samples[train_ix]
val_samples = d.samples[val_ix]
test_samples = d.samples[test_ix]

n = length(train_samples)

df = DataFrame(
    :train => train_samples,
    :val => vcat(val_samples, repeat(["nothing"], n - length(val_samples))),
    :test => vcat(test_samples, repeat(["nothing"], n - length(test_samples)))
)
wsave(datadir("split.csv"), df)

df_indexes = DataFrame(
    :train => train_samples,
    :train_ix => train_ix,
    :val => vcat(val_samples, repeat(["nothing"], n - length(val_samples))),
    :val_ix => vcat(val_ix, repeat([-1], n - length(val_samples))),
    :test => vcat(test_samples, repeat(["nothing"], n - length(test_samples))),
    :test_ix => vcat(test_ix, repeat([-1], n - length(test_samples)))
)
wsave(datadir("split_indexes.csv"), df_indexes)

d1 = DataFrame(:train => train_samples)
d2 = DataFrame(:val => val_samples)
d3 = DataFrame(:test => test_samples)

wsave(datadir("train_samples.csv"), d1)
wsave(datadir("val_samples.csv"), d2)
wsave(datadir("test_samples.csv"), d3)


d1_ix = DataFrame(
    :train => train_samples,
    :train_ix => train_ix
)
d2_ix = DataFrame(
    :val => val_samples,
    :val_ix => val_ix
)
d3_ix = DataFrame(
    :test => test_samples,
    :test_ix => test_ix
)

wsave(datadir("train_samples_indexes.csv"), d1_ix)
wsave(datadir("val_samples_indexes.csv"), d2_ix)
wsave(datadir("test_samples_indexes.csv"), d3_ix)

### DateTime split
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

n = length(train_samples)

df = DataFrame(
    :train => train_samples,
    :val => vcat(val_samples, repeat(["nothing"], n - length(val_samples))),
    :test => vcat(test_samples, repeat(["nothing"], n - length(test_samples)))
)
wsave(splitdir("timesplit_41", "split.csv"), df)

df_indexes = DataFrame(
    :train => train_samples,
    :train_ix => train_ix,
    :val => vcat(val_samples, repeat(["nothing"], n - length(val_samples))),
    :val_ix => vcat(val_ix, repeat([-1], n - length(val_samples))),
    :test => vcat(test_samples, repeat(["nothing"], n - length(test_samples))),
    :test_ix => vcat(test_ix, repeat([-1], n - length(test_samples)))
)
wsave(splitdir("timesplit_41", "split_indexes.csv"), df_indexes)

d1 = DataFrame(:train => train_samples)
d2 = DataFrame(:val => val_samples)
d3 = DataFrame(:test => test_samples)

wsave(splitdir("timesplit_41", "train_samples.csv"), d1)
wsave(splitdir("timesplit_41", "val_samples.csv"), d2)
wsave(splitdir("timesplit_41", "test_samples.csv"), d3)


d1_ix = DataFrame(
    :train => train_samples,
    :train_ix => train_ix
)
d2_ix = DataFrame(
    :val => val_samples,
    :val_ix => val_ix
)
d3_ix = DataFrame(
    :test => test_samples,
    :test_ix => test_ix
)

wsave(splitdir("timesplit_41", "train_samples_indexes.csv"), d1_ix)
wsave(splitdir("timesplit_41", "val_samples_indexes.csv"), d2_ix)
wsave(splitdir("timesplit_41", "test_samples_indexes.csv"), d3_ix)

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

n = length(train_samples)

df = DataFrame(
    :train => train_samples,
    :val => vcat(val_samples, repeat(["nothing"], n - length(val_samples))),
    :test => vcat(test_samples, repeat(["nothing"], n - length(test_samples)))
)
wsave(splitdir("timesplit_57", "split.csv"), df)

df_indexes = DataFrame(
    :train => train_samples,
    :train_ix => train_ix,
    :val => vcat(val_samples, repeat(["nothing"], n - length(val_samples))),
    :val_ix => vcat(val_ix, repeat([-1], n - length(val_samples))),
    :test => vcat(test_samples, repeat(["nothing"], n - length(test_samples))),
    :test_ix => vcat(test_ix, repeat([-1], n - length(test_samples)))
)
wsave(splitdir("timesplit_57", "split_indexes.csv"), df_indexes)

d1 = DataFrame(:train => train_samples)
d2 = DataFrame(:val => val_samples)
d3 = DataFrame(:test => test_samples)

wsave(splitdir("timesplit_57", "train_samples.csv"), d1)
wsave(splitdir("timesplit_57", "val_samples.csv"), d2)
wsave(splitdir("timesplit_57", "test_samples.csv"), d3)


d1_ix = DataFrame(
    :train => train_samples,
    :train_ix => train_ix
)
d2_ix = DataFrame(
    :val => val_samples,
    :val_ix => val_ix
)
d3_ix = DataFrame(
    :test => test_samples,
    :test_ix => test_ix
)

wsave(splitdir("timesplit_57", "train_samples_indexes.csv"), d1_ix)
wsave(splitdir("timesplit_57", "val_samples_indexes.csv"), d2_ix)
wsave(splitdir("timesplit_57", "test_samples_indexes.csv"), d3_ix)