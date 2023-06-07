using Cuckoo
using DrWatson
using Flux
using Flux: @epochs
using StatsBase
using DataFrames

# get the passed arguments
modelname = ARGS[1] # modelname = "pedro"
feature_file = ARGS[2] # feature_file = "/mnt/data/jsonlearning/experiments/cuckoo_small/combined.csv"
seed = parse(Int, ARGS[3]) # seed = 1
rep = parse(Int, ARGS[4]) # rep = 1
ratio = "timesplit"
dataset = "cuckoo"

# load labels file, merge tables and get train/validation/split
tr_x, tr_l, tr_h, val_x, val_l, val_h, test_x, test_l, test_h = load_split_features(
    dataset, feature_file,
    seed=seed,
    ratio=ratio
)
const labelnames = sort(unique(tr_l))
@info "Data loaded."

using Random
# fix seed to always choose the same hyperparameters
function sample_params(seed)
    Random.seed!(seed)

    hdim = sample([8,16,32,64,128,256])
    activation = sample(["sigmoid", "tanh", "relu", "swish"])
    nlayers = sample(2:4)
    batchsize = sample([32,64,128,256])
    transformation = sample(["none", "log", "minmax", "standard"])

    Random.seed!()
    return (hdim=hdim, activation=activation, nlayers=nlayers, batchsize=batchsize, transformation=transformation)
end

# initialize the parameters
idim = size(tr_x, 1)
odim = 10
p = sample_params(rep)
act = eval(Symbol(p.activation))

# create a neural network classifier
if p.nlayers == 2
    model = Chain(Dense(idim, p.hdim, act), Dense(p.hdim, odim))
elseif p.nlayers == 3
    model = Chain(Dense(idim, p.hdim, act), Dense(p.hdim, p.hdim, act), Dense(p.hdim, odim))
else
    model = Chain(Dense(idim, p.hdim, act), Dense(p.hdim, p.hdim, act), Dense(p.hdim, p.hdim, act), Dense(p.hdim, odim))
end
@info "Model created."

# initialize loss, optimizer, model parameters
loss(x, y) = Flux.logitcrossentropy(model(x), y)
opt = ADAM(1e-4)
ps = Flux.params(model)

### Data preparation ###
using Cuckoo: transform_data
tr_x_norm, val_x_norm, test_x_norm = transform_data(tr_x, val_x, test_x, type=p.transformation)
tr_x_norm, val_x_norm, test_x_norm = Float32.(tr_x_norm), Float32.(val_x_norm), Float32.(test_x_norm)
tr_y = Flux.onehotbatch(tr_l, labelnames)
val_y = Flux.onehotbatch(val_l, labelnames)
train_data = Flux.Data.DataLoader((tr_x_norm, tr_y), batchsize=p.batchsize)

# Model training
max_train_time = 60*15 # 15 minutes training time, no early stopping for now

@info "Starting training."
start_time = time()
while time() - start_time < max_train_time
    Flux.train!(loss, ps, train_data, opt)

    train_acc = mean(Flux.onecold(model(tr_x_norm), labelnames) .== tr_l)
    val_acc = mean(Flux.onecold(model(val_x_norm), labelnames) .== val_l)
    test_acc = mean(Flux.onecold(model(test_x_norm), labelnames) .== test_l)

    @info "Train accuracy = $(round(train_acc, digits=3))"
    @info "Validation accuracy = $(round(val_acc, digits=3))"
    @info "Test accuracy = $(round(test_acc, digits=3))"
end

######################
### Saving results ###
######################

using UUIDs
id = "$(uuid1())"   # generate unique uuid to save data in two files - one contains metadata, the other contains the predictions

par_dict = Dict(keys(p) .=> values(p))
info_dict = Dict(
    :uuid => id,
    :feature_model => modelname,
    :classification_model => "dense_classifier",
    :seed => seed,
    :repetition => rep,
    :ratio => ratio,
    :model => model
)

results_dict = merge(par_dict, info_dict)

predictions = vcat(
    encode_labels(Flux.onecold(model(tr_x_norm), labelnames), labelnames),
    encode_labels(Flux.onecold(model(val_x_norm), labelnames), labelnames),
    encode_labels(Flux.onecold(model(test_x_norm), labelnames), labelnames)
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

# add the softmax output
# not currently saved in the dataframe
# softmax_output = DataFrame(
#     hcat(
#         softmax(model(tr_x_norm)),
#         softmax(model(val_x_norm)),
#         softmax(model(test_x_norm))
#     ) |> transpose |> collect, :auto
# )

# final_df = hcat(results_df, softmax_output)

@info "Results calculated."

"""
Name of the model is used to create a folder containing both the features and results of the model.
Each subfolder is named by a neural network model acting on those features.

There are two files saved for each model: bson file and csv file:
- bson file contains metadata: parameters of features, model, seeds etc.
- csv file contains results: hash, ground truth, predicted labels, split names, and softmax output
"""

using Cuckoo: cuckoo_hash_to_int
df = cuckoo_hash_to_int(results_df)

@info "Saving results..."
safesave(expdir("results", modelname, "dense_classifier", "$id.bson"), results_dict)
# safesave(expdir("results", modelname, "dense_classifier", "$id.csv"), results_df)
safesave(expdir("results", modelname, "dense_classifier", "$id.csv"), df)
@info "Results saved, experiment finished."