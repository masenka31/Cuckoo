using DrWatson
using Cuckoo
using CSV, DataFrames
using Statistics
using ProgressMeter
using PrettyTables
using Mill, Flux

accuracy(y1::T, y2::T) where T = mean(y1 .== y2)

function load_results(modelname::String="hmil", dataset::String="garcia")
    resultspath = expdir("results", "garcia", modelname, "dense_classifier")
    bson_results = collect_results(resultspath)
    ids = bson_results.uuid

    results = []

    for (i, id) in enumerate(ids)
        # file = expdir("cuckoo_small", modelname, "dense_classifier", "$id.csv")
        file = expdir("results", modelname, "dense_classifier", "$id.csv")
        # file = expdir("results", "garcia", modelname, "dense_classifier", "$id.csv")
        df = CSV.read(file, DataFrame)

        # replace String3 with 0, 1 and missing
        function replace_me(x)
            if x == "1"
                return 1
            elseif x == "0"
                return 0
            else
                return missing
            end
        end
        df.predicted_parsed = replace_me.(df.predicted)
        df = df[:, Not(:predicted)]
        
        rename!(df, :predicted_parsed => :predicted)
        filter!(:predicted => x -> !ismissing(x), df)
        df[!, :predicted] = convert.(Int64, df.predicted)
        df[!, :predicted] .+= 1

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
    par_d = bson_results[1, :]

    groupkeys = names(DataFrame(par_d)[:, Not([:repetition, :uuid, :model, :feature_model, :path, :seed])])
    gg = groupby(results, groupkeys)
    gg = filter(x -> nrow(x) == 5, gg)
    cdf = combine(gg, [:train, :validation, :test] .=> mean, keepkeys=true, renamecols=false)
    sort!(cdf, :validation, rev=true)
    return cdf
end

function load_results(modelname::String; show=true)
    resultspath = expdir("results", modelname, "dense_classifier", "timesplit")
    # resultspath = expdir("results", "garcia", modelname, "dense_classifier")
    # bson_results = collect_results(resultspath)
    bson_results = collect_results(resultspath, rexclude=[r"0318210e.*"])
    ids = bson_results.uuid

    results = []

    for (i, id) in enumerate(ids)
        # file = expdir("cuckoo_small", modelname, "dense_classifier", "$id.csv")
        file = expdir("results", modelname, "dense_classifier", "timesplit", "$id.csv")
        # file = expdir("results", "garcia", modelname, "dense_classifier", "$id.csv")
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

    if modelname == "hmil"
        groupkeys = ["repetition", "activation", "nlayers", "batchsize", "aggregation", "mdim", "tr_ratio"]
    elseif modelname in ["pedro007", "vasek007"]
        groupkeys = ["repetition", "activation", "nlayers", "batchsize", "hdim", "tr_ratio"]
    end
    gg = groupby(results, groupkeys)
    b = map(x -> nrow(gg[x]) > 4, 1:length(gg))
    gg = gg[b]
    cdf = combine(gg, [:train, :validation, :test] .=> mean, keepkeys=true, renamecols=false)

    sort!(cdf, :validation, rev=true)
    tr_g = groupby(cdf, :tr_ratio)
    if show
        foreach(x -> pretty_table(tr_g[x][1:5, :]), 1:length(tr_g))
    end
    return tr_g
end
