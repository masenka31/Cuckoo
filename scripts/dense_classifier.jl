using DrWatson
using Flux
using Flux: @epochs

include(srcdir("paths.jl"))
include(srcdir("dataset.jl"))
include(srcdir("data.jl"))
include(srcdir("utils.jl"))

modelname = ARGS[1]
feature_file = ARGS[2]
seed = parse(Int, ARGS[3])
rep = parse(Int, ARGS[4])
labels_file = datadir("labels.csv")
df = CSV.read(labels_file, DataFrame)
tr_ratio = 60

tr_x, tr_l, tr_h, val_x, val_l, val_h, test_x, test_l, test_h = load_split_features(
    feature_file, labels_file,
    seed=seed,
    tr_ratio=tr_ratio
)
const labelnames = sort(unique(df.family))
@info "Data loaded."

using Random
# fix seed to always choose the same hyperparameters
function sample_params(seed)
    Random.seed!(seed)

    hdim = sample([8,16,32,64,128,256])
    activation = sample(["sigmoid", "tanh", "relu", "swish"])
    nlayers = sample(2:4)
    batchsize = sample([32,64,128,256])

    Random.seed!()
    return (hdim=hdim, activation=activation, nlayers=nlayers, batchsize=batchsize)
end

idim = size(tr_x, 1)
odim = 10

p = sample_params(rep)
act = eval(Symbol(p.activation))

if p.nlayers == 2
    model = Chain(Dense(idim, p.hdim, act), Dense(p.hdim, odim))
elseif p.nlayers == 3
    model = Chain(Dense(idim, p.hdim, act), Dense(p.hdim, p.hdim, act), Dense(p.hdim, odim))
else
    model = Chain(Dense(idim, p.hdim, act), Dense(p.hdim, p.hdim, act), Dense(p.hdim, p.hdim, act), Dense(p.hdim, odim))
end

@info "Model created."

loss(x, y) = Flux.logitcrossentropy(model(x), y)
opt = ADAM()
ps = Flux.params(model)

tr_y = Flux.onehotbatch(tr_l, labelnames)
val_y = Flux.onehotbatch(val_l, labelnames)
train_data = Flux.Data.DataLoader((tr_x, tr_y), batchsize=p.batchsize)

@info "Starting training."
start_time = time()
max_train_time = 60*60 # hour training time, no early stopping for now
while time() - start_time < max_train_time
    Flux.train!(loss, ps, train_data, opt)
    acc = loss(val_x, val_y)
    @info "validation accuracy = $(round(acc, digits=3))"
end

using UUIDs
id = "$(uuid1())"

par_dict = Dict(keys(p) .=> values(p))
info_dict = Dict(
    :uuid => id,
    :feature_model => modelname,
    :nn_model => "dense_classifier",
    :seed => seed,
    :repetition => rep,
    :tr_ratio => tr_ratio
)

results_dict = merge(par_dict, info_dict)

# calculate predicted labels
predictions = vcat(
    Flux.onecold(model(tr_x), labelnames),
    Flux.onecold(model(val_x), labelnames),
    Flux.onecold(model(test_x), labelnames)
)

softmax_output = DataFrame(
    hcat(
        model(tr_x),
        model(val_x),
        model(test_x)
    ) |> transpose |> collect, :auto)

# add softmax output
results_df = DataFrame(
    :hash => vcat(tr_h, val_h, test_h),
    :ground_truth => vcat(tr_l, val_l, test_l),
    :predicted => predictions,
    :split => vcat(
        repeat(["train"], length(tr_h)),
        repeat(["validation"], length(val_h)),
        repeat(["test"], length(test_h))
    )
)

final_df = hcat(results_df, softmax_output)
@info "Results calculated."

"""
Name of the model is used to create a folder containing both the features and results of the model.
Each subfolder is named by a neural network model acting on those features.

There are two files saved for each model: bson file and csv file:
- bson file contains metadata: parameters of features, model, seeds etc.
- csv file contains results: hash, ground truth, predicted labels, split names, and softmax output
"""
@info "Saving results"
safesave(expdir("cuckoo_small", modelname, "dense_classifier", "$id.bson"), results_dict)
safesave(expdir("cuckoo_small", modelname, "dense_classifier", "$id.csv"), results_df)
@info "Results saved, experiment finished."