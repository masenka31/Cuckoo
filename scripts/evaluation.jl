using DrWatson
using DataFrames, CSV
using Statistics
include(srcdir("paths.jl"))

accuracy(y1::T, y2::T) where T = mean(y1 .== y2)

modelname = "pedro007" #ARGS[1]

bson_results = collect_results(expdir("cuckoo_small", modelname, "dense_classifier"))
ids = bson_results.uuid

results = []

for (i, id) in enumerate(ids)
    file = expdir("cuckoo_small", modelname, "dense_classifier", "$id.csv")
    df = CSV.read(file, DataFrame)
    gdf = groupby(df, :split)

    # calculate accuracy
    cdf = combine(gdf, [:ground_truth, :predicted] => accuracy => :accuracy)
    d = DataFrame(cdf.split .=> cdf.accuracy)
    
    # add hyperparameters
    par_d = bson_results[i, :]
    d_end = hcat(d, DataFrame(par_d))

    push!(results, d_end)
end

results = vcat(results...)

# groupby
# it should be possible to only group on repetition and tr_ratio to get the groups
# time split does not have multiple seeds

if modelname == "hmill_classifier"
    groupkeys = ["repetition", "activation", "nlayers", "batchsize", "aggregation", "mdim", "tr_ratio"]
else
    groupkeys = ["repetition", "activation", "nlayers", "batchsize", "hdim", "tr_ratio"]
end
gg = groupby(results, groupkeys)
cdf = combine(gg, [:train, :validation, :test] .=> mean, keepkeys=true, renamecols=false)

sort!(cdf, :validation, rev=true)
tr_g = groupby(cdf, :tr_ratio)
foreach(x -> pretty_table(tr_g[x]), 1:length(tr_g))