using DrWatson
using Cuckoo
using JsonGrinder
using Mill
using StatsBase
using DataFrames
using Flux
using UUIDs
using ProgressMeter

# load data and split into train/validation/test
d = Dataset("cuckoo")
labels = d.family
const labelnames = sort(unique(labels))

# specify the seed and type of the split
seed = 1
tr_ratio = "timesplit"

# loads data into ProductNode format
@time tr_x, tr_l, tr_h, val_x, val_l, val_h, test_x, test_l, test_h = load_split_indexes(
    d, seed=seed, tr_ratio=tr_ratio
);

function minibatch(X::ProductNode, y; batchsize=64, labelnames=labelnames)
    # sample random indexes
    ix = sample(1:length(y), batchsize)
    # index into the data with the sampled indexes
    x, y = X[ix], y[ix]
    # convert labels to onehotbatches
    yoh = Flux.onehotbatch(y, labelnames)

    return (x, yoh)
end

using Random
# fix seed to always choose the same hyperparameters
function sample_params(seed)
    Random.seed!(seed)

    # possible parameter values
    mdim = sample([8,16,32,64,128,256])
    activation = sample(["sigmoid", "tanh", "relu", "swish"])
    nlayers = sample(2:4)
    batchsize = sample([32,64,128,256])
    aggregation = sample(["SegmentedMeanMax", "SegmentedMax", "SegmentedMean"])

    Random.seed!()
    return (mdim=mdim, activation=activation, nlayers=nlayers, batchsize=batchsize, aggregation=aggregation)
end

parameters = sample_params(31)

xtr, ytr = minibatch(tr_x, tr_l, batchsize=2)
full_model = classifier_constructor(xtr; parameters..., odim=10);

mill_model = reflectinmodel(
    xtr[1],
    k -> Dense(k, 32, swish),
    # d -> BagCount ∘ SegmentedMeanMax(d)
    d -> SegmentedMeanMax(d)
);
# net = Dense(32, 10)
net = Chain(Dense(32, 32, relu), Dense(32, 10))
model = Chain(mill_model, net);

mill_model(xtr)
net(mill_model(xtr))
model(xtr)

loss(X, y) = Flux.logitcrossentropy(model(X), y)
loss(xtr, ytr)

# setup optimizer
opt = ADAM(1e-4)

# training - first pass takes very long
@time Flux.train!(loss, Flux.params(model), [(xtr, ytr)], opt)

# train for specified time
max_train_time = 60 # seconds

# time training loop
start_time = time()
while time() - start_time < max_train_time
    batches = map(_ -> minibatch(tr_x, tr_l), 1:5); # samples 5 minibatches
    Flux.train!(loss, Flux.params(model), batches, opt) # train on those minibatches
    batch_loss = mean(x -> loss(x...), batches) # calculate the loss
    @info "batch loss = $(round(batch_loss, digits=3))" # print the loss
end

# no epochs training loop§
for epoch in 1:100
    batches = map(_ -> minibatch(tr_x, tr_l), 1:5); # samples 5 minibatches
    Flux.train!(loss, Flux.params(model), batches, opt) # train on those minibatches
    batch_loss = mean(x -> loss(x...), batches) # calculate the loss
    print("Epoch: $epoch")
    @info "batch loss = $(round(batch_loss, digits=3))" # print the loss
end

# calculate results
probabilities = model(xtr)
predicted_labels = Flux.onecold(probabilities, labelnames)
ytr # onehot

probabilities = model(tr_x[1:10])
predicted_labels = Flux.onecold(probabilities, labelnames)
# look at
#  predicted_labels vs tr_l[1:10]
using Statistics
accucary = mean(predicted_labels .== tr_l[1:10])

