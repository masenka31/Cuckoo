using DrWatson
using Cuckoo
using XGBoost
using Statistics, StatsBase
using DataFrames

# get the passed arguments
if isempty(ARGS)
    modelname = "pedro"
    feature_file = "/mnt/data/jsonlearning/experiments/feature_vectors/pedro_garcia.csv"
    seed = 1
    rep = 1
else
    modelname = ARGS[1] # modelname = "pedro"
    feature_file = ARGS[2] # feature_file = "/mnt/data/jsonlearning/experiments/feature_vectors/pedro_garcia.csv"
    seed = parse(Int, ARGS[3]) # seed = 1
    rep = parse(Int, ARGS[4]) # rep = 1
end
dataset = "garcia"

# load labels file, merge tables and get train/validation/split
tr_x, tr_l, tr_h, val_x, val_l, val_h, test_x, test_l, test_h = load_split_features(
    dataset, feature_file,
    seed=seed,
    ratio=dataset
)
const labelnames = sort(unique(tr_l))
@info "Data loaded."

# prepare data for XGBoost

X = collect(tr_x')
y = tr_l .== labelnames[2]

Xval = collect(val_x')
yval = val_l .== labelnames[2]

Xtest = collect(test_x')
ytest = test_l .== labelnames[2]

# parameters
using Random
# fix seed to always choose the same hyperparameters
function sample_params(seed)
    Random.seed!(seed)

    max_depth = sample(5:5:40)
    eta = sample([0, 0.3, 0.5, 0.7, 1])
    # gamma
    min_child_weight = sample([0, 1, 2])
    subsample = sample([0.25, 0.5, 0.75, 1])

    Random.seed!()
    return (max_depth=max_depth, eta=eta, min_child_weight=min_child_weight, subsample=subsample)
end

p = sample_params(rep)
model = xgboost(
    (X, y), num_round=500,
    max_depth=p.max_depth, eta=p.eta, min_child_weight=p.min_child_weight, subsample=p.subsample,
    objective="binary:logistic"
)

ypred = XGBoost.predict(model, Xval) .> 0.5
av = mean(ypred .== yval)
ypred = XGBoost.predict(model, Xtest) .> 0.5
at = mean(ypred .== ytest)
println("Val accuracy: $(round(av, digits=4))")
println("Test accuracy: $(round(at, digits=4))")

######################
### Saving results ###
######################

using UUIDs
id = "$(uuid1())"   # generate unique uuid to save data in two files - one contains metadata, the other contains the predictions

par_dict = Dict(keys(p) .=> values(p))
info_dict = Dict(
    :uuid => id,
    :feature_model => modelname,
    :classification_model => "xgboost",
    :seed => seed,
    :model => model
)

results_dict = merge(par_dict, info_dict)

predictions = vcat(
    (XGBoost.predict(model, X) .> 0.5) .|> Int8,
    (XGBoost.predict(model, Xval) .> 0.5) .|> Int8,
    (XGBoost.predict(model, Xtest) .> 0.5) .|> Int8,
)

# predictions of the model
results_df = DataFrame(
    :hash => vcat(tr_h, val_h, test_h),
    :ground_truth => vcat(
        encode_labels(tr_l, labelnames),
        encode_labels(val_l, labelnames),
        encode_labels(test_l, labelnames)
    ),
    :predicted => predictions,
    :split => Int8.(vcat(
        repeat([0], length(tr_h)),
        repeat([1], length(val_h)),
        repeat([2], length(test_h))
    ))
)

@info "Results calculated."

"""
Name of the model is used to create a folder containing both the features and results of the model.
Each subfolder is named by a neural network model acting on those features.

There are two files saved for each model: bson file and csv file:
- bson file contains metadata: parameters of features, model, seeds etc.
- csv file contains results: hash, ground truth, predicted labels, split names, and softmax output
"""

using Cuckoo: garcia_hash_to_int
df = garcia_hash_to_int(results_df)

@info "Saving results..."
safesave(expdir("results", dataset, modelname, "xgboost", "$id.bson"), results_dict)
safesave(expdir("results", dataset, modelname, "xgboost", "$id.csv"), df)
@info "Results saved, experiment finished."
