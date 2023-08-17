using DrWatson
using Cuckoo
using JsonGrinder
using Mill
using StatsBase
using DataFrames
using Flux
using UUIDs
using ProgressMeter
using BSON
using CSV

# get the passed arguments
modelname = "hmil"
tr_ratio = "timesplit"
dataset = "cuckoo_small"

# load data and split into train/validation/test
d = Dataset("cuckoo_small")
hash = map(x -> split(split(x, '/')[end], '.')[1], d.samples)
dates = d.date

date_df = DataFrame(:hash => hash, :date => dates, :path => d.samples)
sort!(date_df, :date)

#################################################################################
###                                  Testing                                  ###
#################################################################################

feature_model = "pedro"
end_model = "dense_classifier"
results_dir = expdir("results", feature_model, end_model) # results directory
files = readdir(results_dir)
ids = unique([x[1:end-5] for x in files if endswith(x, ".bson")])
id = ids[3]

# load the results and calculate metrics
df = CSV.read(joinpath(results_dir, "$id.csv"), DataFrame)
if "int_hash" in names(df)
    df = Cuckoo.cuckoo_int_to_hash(df)
end

r = innerjoin(date_df, df, on=:hash)
filter!(:date => x -> x > crop_date, r)
df = r


# load data and split into train/validation/test
d = Dataset(dataset)

hash = map(x -> split(split(x, '/')[end], '.')[1], d.samples)
dates = d.date

df = DataFrame(:hash => hash, :date => dates, :path => d.samples)
sort!(df, :date)

feature_model = "hmil"
end_model = "dense_classifier"
results_dir = expdir("results", dataset, feature_model, end_model) # results directory
files = readdir(results_dir)
ids = unique([x[1:end-5] for x in files if endswith(x, ".bson")])

id = ids[3]

using BSON
# load the results and calculate metrics
c = CSV.read(joinpath(results_dir, "$id.csv"), DataFrame)

### With last value from train/validation

r = innerjoin(df, c, on=:hash)
sort!(r, :date)
# r = r[100:end, :]
train_val = filter(:split => x -> x != "test", r)
filter!(:split => x -> x == "test", r)

gdf = groupby(r, :date)
n_samples = []
cum_acc = []

push!(cum_acc, mean(train_val.ground_truth .== train_val.predicted))

concat_df = gdf[1]
acc = mean(concat_df.ground_truth .== concat_df.predicted)
push!(cum_acc, acc)
push!(n_samples, nrow(concat_df))

for i in 2:length(gdf)
    concat_df = vcat(concat_df, gdf[i])
    acc = mean(concat_df.ground_truth .== concat_df.predicted)
    push!(cum_acc, acc)
    push!(n_samples, nrow(concat_df))
end

first_ix = findfirst(x -> x == "test", r.split)
first_date = r.date[first_ix]

unq_dates = unique(r.date)
unq_dates = vcat(train_val.date[end], unq_dates)

plot(unq_dates, cum_acc)
plot!([first_date, first_date], [minimum(cum_acc),1], color=:black, ylim=(minimum(cum_acc) - 0.01, 1))
savefig("plot.png")

### ALL

r = innerjoin(df, c, on=:hash)
sort!(r, :date)
r = r[100:end, :]

gdf = groupby(r, :date)
n_samples = []
cum_acc = []

concat_df = gdf[1]
acc = mean(concat_df.ground_truth .== concat_df.predicted)
push!(cum_acc, acc)
push!(n_samples, nrow(concat_df))

for i in 2:length(gdf)
    concat_df = vcat(concat_df, gdf[i])
    acc = mean(concat_df.ground_truth .== concat_df.predicted)
    push!(cum_acc, acc)
    push!(n_samples, nrow(concat_df))
end

first_ix = findfirst(x -> x == "test", r.split)
first_date = r.date[first_ix]

unq_dates = unique(r.date)

plot(unq_dates, cum_acc)
plot!([first_date, first_date], [minimum(cum_acc),1], color=:black, ylim=(minimum(cum_acc) - 0.01, 1))
savefig("plot.png")

#################################################################################
###                              Monthly changes                              ###
#################################################################################

import Cuckoo: _calculate_accuracy_date, _calculate_accuracy
using Cuckoo: accuracy

R = DataFrame[]

@showprogress for id in ids
    # load parameters
    b = BSON.load(joinpath(results_dir, "$id.bson"))
    # delete model from the file (too large)
    delete!(b, :model)
    # create a parameters dataframe from it
    bdf = DataFrame(b)

    df = CSV.read(joinpath(results_dir, "$id.csv"), DataFrame)
    # _calculate_accuracy_date(df, date_df, Date(2019, 1, 1), "after")
    # _calculate_accuracy_date(df, date_df, Date(2019, 7, 1), "before")
    
    train_acc, validation_acc, test_acc = _calculate_accuracy(df)
    bdf[!, :train_acc] = [train_acc]
    bdf[!, :validation_acc] = [validation_acc]
    bdf[!, :test_acc] = [test_acc]

    for m in 7:12
        train_acc, validation_acc, test_acc = _calculate_accuracy_date(df, date_df, Date(2019, m, 1), crop="before")
        bdf[!, Symbol("test_cum_$m")] = [test_acc]
        if m < 12
            train_acc, validation_acc, test_acc = _calculate_accuracy_date(df, date_df, Date(2019, m, 1), crop="between", second_crop_date=Date(2019, m+1, 1))
            bdf[!, Symbol("test_$m")] = [test_acc]
        end
    end

    push!(R, bdf)
end

result = vcat(R...)
sort!(result, :test_acc, rev=true)
first(result, 5)

filtered_columns = ["feature_model", "nn_model", "seed", "uuid", "train_acc", "validation_acc", "test_acc",
                    "test_cum_month", "test_month"]
other_fc = ["test_$m" for m in 7:11]
other_fc2 = ["test_cum_$m" for m in 7:12]
filtered_columns = vcat(filtered_columns, other_fc, other_fc2)


n = names(result)
p = n[[all(ni .!= filtered_columns) for ni in n]]

gdf = groupby(result, p)
gdf = filter(g -> nrow(g) >= 5, gdf)

vals = filter(x -> contains(x, "test") || contains(x, "train") || contains(x, "validation"), names(result))

cdf = combine(gdf, vals .=> mean, renamecols=false)
sort!(cdf, :test_acc, rev=true)

include(srcdir("plotting.jl"))

cum_plot = plot()
monthly_plot = plot()

for r in eachrow(cdf)
    cum_values = r[other_fc2] |> Array
    monthly_values = r[other_fc] |> Array

    cum_plot = plot!(cum_plot, cum_values, label="", color=:grey, lw=0.5)
    monthly_plot = plot!(monthly_plot, monthly_values, label="", color=:grey, l2=0.5)
end

save("plot.png", cum_plot)
save("plot.png", monthly_plot)

function parse_string3(x)
    if x == "0"
        return 1
    elseif x == "1"
        return 2
    else
        return 3
    end
end

data[3]

