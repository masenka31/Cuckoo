using Cuckoo
using JsonGrinder, JSON
using Mill
using CSV, DataFrames
using DrWatson

d = Dataset("garcia")

seed = 1
split_df = CSV.read(splitsdir("garcia/0$(seed)_split.csv"), DataFrame)
hash = d.samples
data_df = DataFrame(
    :i => 1:length(d.family),
    :hash => hash,
)
df = innerjoin(data_df, split_df, on=:hash)

train_ix = filter(:split => x -> x == "train", df)[!, :i]
val_ix = filter(:split => x -> x == "validation", df)[!, :i]
test_ix = filter(:split => x -> x == "test", df)[!, :i]

Xtest, ytest, mtest = d[test_ix]
