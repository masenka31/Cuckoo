using DrWatson
using CSV, DataFrames
using Statistics
using ProgressMeter
using PrettyTables

accuracy(y1::T, y2::T) where T = mean(y1 .== y2)

function load_results(modelname::String; show=true)
    resultspath = expdir("results", modelname, "dense_classifier")
    bson_results = collect_results(resultspath)
    ids = bson_results.uuid

    results = []

    for (i, id) in enumerate(ids)
        # file = expdir("cuckoo_small", modelname, "dense_classifier", "$id.csv")
        file = expdir("results", modelname, "dense_classifier", "$id.csv")
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

    if modelname == "hmill_classifier"
        groupkeys = ["repetition", "activation", "nlayers", "batchsize", "aggregation", "mdim", "tr_ratio"]
    elseif modelname in ["pedro007", "vasek007"]
        groupkeys = ["repetition", "activation", "nlayers", "batchsize", "hdim", "tr_ratio"]
    end
    gg = groupby(results, groupkeys)
    cdf = combine(gg, [:train, :validation, :test] .=> mean, keepkeys=true, renamecols=false)

    sort!(cdf, :validation, rev=true)
    tr_g = groupby(cdf, :tr_ratio)
    if show
        foreach(x -> pretty_table(tr_g[x][1:5, :]), 1:length(tr_g))
    end
    return tr_g
end
