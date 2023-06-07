using DrWatson
using Cuckoo
using StatsBase
using ProgressMeter
using PrettyTables
using Mill, Flux
using BSON, CSV, DataFrames
using XGBoost

accuracy(y1::T, y2::T) where T = mean(y1 .== y2)

"""
    get_results(dataset::String="cuckoo_small", feature_model::String="hmil", end_model::String="dense_classifier")

Loads the results based on the parameters. Does NOT do any aggregation of the results over seeds.

**Parameters**:
- `dataset`: name of the dataset
- `feature_model`: the model used to create features
    - hmil
    - pedro007
    - vasek007
- `end_model`: classifier on top of the features
    - dense_classifier
    - xgboost
"""
function get_results(dataset::String="cuckoo_small", feature_model::String="hmil", end_model::String="dense_classifier")
    # get results bson files
    results_dir = expdir("results", dataset, feature_model, end_model)
    files = readdir(results_dir)
    ids = unique([x[1:end-5] for x in files if endswith(x, ".bson")])

    R = []
    @showprogress for id in ids
        try
            # load parameters
            b = BSON.load(joinpath(results_dir, "$id.bson"))
            # delete model from the file (too large)
            delete!(b, :model)
            # create a parameters dataframe from it
            bdf = DataFrame(b)

            # load the results and calculate metrics
            c = CSV.read(joinpath(results_dir, "$id.csv"), DataFrame)
            train_acc, validation_acc, test_acc = _calculate_accuracy(c) # accuracy
            
            # add accuracy to the dataframe (more metrics can be added if necessary)
            bdf[!, :train_acc] = [train_acc]
            bdf[!, :validation_acc] = [validation_acc]
            bdf[!, :test_acc] = [test_acc]

            push!(R, bdf)
        catch e
            @warn "File with id $id corrupted, skipping loading..."
        end
    end
    vcat(R...)
end

filtered_columns = ["feature_model", "nn_model", "seed", "uuid", "train_acc", "validation_acc", "test_acc"]

"""
    get_combined_results(dataset::String="cuckoo_small", feature_model::String="hmil", end_model::String="dense_classifier"; filtered_columns=filtered_columns)

Loads the results based on the parameters, aggregates them over seeds and returns a sorted
dataframe with the parameters and final metrics.

**Parameters**:
- `dataset`: name of the dataset
- `feature_model`: the model used to create features
    - hmil
    - pedro077
    - vasek007
- `end_model`: classifier on top of the features
    - dense_classifier

**Kwargs**:
- `filtered_columns`: filters out the column names that are not used for aggregation (all columns which are not model parameters have to be put inside this array to make the averaging correct)
"""
function get_combined_results(dataset::String="cuckoo_small", feature_model::String="hmil", end_model::String="dense_classifier"; filtered_columns=filtered_columns, min_seed::Int=5)
    df = get_results(dataset, feature_model, end_model)
    n = names(df)
    p = n[[all(ni .!= filtered_columns) for ni in n]]

    gdf = groupby(df, p)
    gdf = filter(g -> nrow(g) >= min_seed, gdf)
    cdf = combine(gdf, [:train_acc, :validation_acc, :test_acc] .=> mean, renamecols=false)
    return sort(cdf, :validation_acc, rev=true)
end

"""
    _calculate_accuracy(df::DataFrame)

Uses the column `split` to calculate accuracy on train / validation / test splits based on columns
`ground_truth` and `predicted`. Returns the 3-tuple of accuracy values.
"""
function _calculate_accuracy(df::DataFrame)
    if "train" in unique(df.split)
        train = filter(:split => x -> x == "train", df)
        validation = filter(:split => x -> x == "validation", df)
        test = filter(:split => x -> x == "test", df)
    elseif 0 in unique(df.split)
        train = filter(:split => x -> x == 0, df)
        validation = filter(:split => x -> x == 1, df)
        test = filter(:split => x -> x == 2, df)
    end

    train_acc = accuracy(train.ground_truth, train.predicted)
    validation_acc = accuracy(validation.ground_truth, validation.predicted)
    test_acc = accuracy(test.ground_truth, test.predicted)

    return train_acc, validation_acc, test_acc
end

# function load_results(modelname::String="hmil", dataset::String="garcia")
#     resultspath = expdir("results", "garcia", modelname, "dense_classifier")
#     bson_results = collect_results(resultspath)
#     ids = bson_results.uuid

#     results = []

#     for (i, id) in enumerate(ids)
#         # file = expdir("cuckoo_small", modelname, "dense_classifier", "$id.csv")
#         # file = expdir("results", modelname, "dense_classifier", "$id.csv")
#         file = expdir("results", "garcia", modelname, "dense_classifier", "$id.csv")
#         df = CSV.read(file, DataFrame)

#         # replace String3 with 0, 1 and missing
#         function replace_me(x)
#             if x == "1"
#                 return 1
#             elseif x == "0"
#                 return 0
#             else
#                 return missing
#             end
#         end
#         df.predicted_parsed = replace_me.(df.predicted)
#         df = df[:, Not(:predicted)]
        
#         rename!(df, :predicted_parsed => :predicted)
#         filter!(:predicted => x -> !ismissing(x), df)
#         df[!, :predicted] = convert.(Int64, df.predicted)
#         df[!, :predicted] .+= 1

#         gdf = groupby(df, :split)

#         # calculate accuracy
#         cdf = combine(gdf, [:ground_truth, :predicted] => accuracy => :accuracy)
#         d = DataFrame(cdf.split .=> cdf.accuracy)
        
#         # add hyperparameters
#         par_d = bson_results[i, :]
#         d_end = hcat(d, DataFrame(par_d))

#         push!(results, d_end)
#     end

#     results = vcat(results...)
#     par_d = bson_results[1, :]

#     groupkeys = names(DataFrame(par_d)[:, Not([:repetition, :uuid, :model, :feature_model, :path, :seed])])
#     gg = groupby(results, groupkeys)
#     gg = filter(x -> nrow(x) == 5, gg)
#     cdf = combine(gg, [:train, :validation, :test] .=> mean, keepkeys=true, renamecols=false)
#     sort!(cdf, :validation, rev=true)
#     return cdf
# end

# function load_results(modelname::String; show=true)
#     resultspath = expdir("results", modelname, "dense_classifier", "timesplit")
#     # resultspath = expdir("results", "garcia", modelname, "dense_classifier")
#     # bson_results = collect_results(resultspath)
#     bson_results = collect_results(resultspath, rexclude=[r"0318210e.*"])
#     ids = bson_results.uuid

#     results = []

#     for (i, id) in enumerate(ids)
#         # file = expdir("cuckoo_small", modelname, "dense_classifier", "$id.csv")
#         # file = expdir("results", modelname, "dense_classifier", "timesplit", "$id.csv")
#         file = expdir("results", "garcia", modelname, "dense_classifier", "$id.csv")
#         df = CSV.read(file, DataFrame)
#         gdf = groupby(df, :split)

#         # calculate accuracy
#         cdf = combine(gdf, [:ground_truth, :predicted] => accuracy => :accuracy)
#         d = DataFrame(cdf.split .=> cdf.accuracy)
        
#         # add hyperparameters
#         par_d = bson_results[i, :]
#         d_end = hcat(d, DataFrame(par_d))

#         push!(results, d_end)
#     end

#     results = vcat(results...)

#     if modelname == "hmil"
#         groupkeys = ["repetition", "activation", "nlayers", "batchsize", "aggregation", "mdim", "tr_ratio"]
#     elseif modelname in ["pedro007", "vasek007"]
#         groupkeys = ["repetition", "activation", "nlayers", "batchsize", "hdim", "tr_ratio"]
#     end
#     gg = groupby(results, groupkeys)
#     b = map(x -> nrow(gg[x]) > 4, 1:length(gg))
#     gg = gg[b]
#     cdf = combine(gg, [:train, :validation, :test] .=> mean, keepkeys=true, renamecols=false)

#     sort!(cdf, :validation, rev=true)
#     tr_g = groupby(cdf, :tr_ratio)
#     if show
#         foreach(x -> pretty_table(tr_g[x][1:5, :]), 1:length(tr_g))
#     end
#     return tr_g
# end
