using DrWatson
include(srcdir("dataset.jl"))
include(srcdir("data.jl"))
include(srcdir("constructors.jl"))

using Flux
using Flux: @epochs

# load data
d = Dataset()
labels = d.family
const labelnames = sort(unique(labels))
# data, _ = d[:]

# split data
seed = parse(Int, ARGS[1])
ratios=(0.2,0.4,0.4)
train_ix, val_ix, test_ix = train_val_test_ix(labels; ratios=ratios, seed=seed)
Xtrain, ytrain = d[train_ix]
Xval, yval = d[val_ix]
Xtest, ytest = d[test_ix]

# minibatch function
function minibatch(d::Dataset, train_ix; batchsize=64)
    ix = sample(train_ix, batchsize)
    labelnames = sort(unique(d.family))
    x, y = d[ix]
    yoh = Flux.onehotbatch(y, labelnames)
    return (x, yoh)
end
function minibatch(x::ProductNode, y; batchsize=64)
    ix = sample(1:length(y), batchsize)
    labelnames = sort(unique(y))
    x, y = d[ix]
    yoh = Flux.onehotbatch(y, labelnames)
    return (x, yoh)
end

##########################################
### Parameters, model, loss & accuracy ###
##########################################

function sample_params()
    mdim = sample([8,16,32,64])
    activation = sample(["sigmoid", "tanh", "relu", "swish"])
    aggregation = sample(["SegmentedMeanMax", "SegmentedMax", "SegmentedMean"])
    nlayers = sample(1:3)
    return (mdim=mdim, activation=activation, aggregation=aggregation, nlayers=nlayers)
end

# Parameters
# mdim, activation, aggregation, nlayers = 32, relu, BagCount ∘ SegmentedMeanMax, 3
parameters = sample_params()
parameters = merge(parameters, (seed = seed, ))

xtr, _ = minibatch(d, train_ix)
full_model = classifier_constructor(xtr; parameters..., odim=length(labelnames));
mill_model = full_model[1];

opt = ADAM()

# loss and accuracy
loss(X, y) = Flux.logitcrossentropy(full_model(X), y)
accuracy(x::ProductNode, y::Vector{String}) = mean(labelnames[Flux.onecold(full_model(x))] .== y)
accuracy(x::ProductNode, yoh::Flux.OneHotMatrix) = mean(labelnames[Flux.onecold(full_model(x))] .== Flux.onecold(yoh, labelnames))
accuracy(y1::Vector{T}, y2::Vector{T}) where T = mean(y1 .== y2)

@info "Model created, starting training..."

# train the model
start_time = time()
max_train_time = 60*10 # 10 minutes of training time
while time() - start_time < max_train_time
    batches = map(_ -> minibatch(d, train_ix), 1:10);
    Flux.train!(loss, Flux.params(full_model), batches, opt)
    # batch_loss = mean(x -> loss(x...), batches)
    # @info "batch loss = $(round(batch_loss, digits=3))"
    val_batch = minibatch(d, test_ix; batchsize=100)
    acc = accuracy(val_batch...)
    @info "validation batch accuracy = $(round(acc, digits=3))"
end

# calculate predicted labels
ztrain = encode_labels(Flux.onecold(full_model(Xtrain), labelnames), labelnames)
zval = encode_labels(Flux.onecold(full_model(Xval), labelnames), labelnames)
ztest = encode_labels(Flux.onecold(full_model(Xtest), labelnames), labelnames)

# save true and predicted labels to a dataframe
train_df = DataFrame(
    :ytrain => encode_labels(ytrain, labelnames),
    :ztrain => ztrain,
)
val_df = DataFrame(
    :yval => encode_labels(yval, labelnames),
    :zval => zval,
)
test_df = DataFrame(
    :ytest => encode_labels(ytest, labelnames),
    :ztest => ztest,
)

# ### one version of saving
# # save the dataframes to csv
# wsave(datadir("classifier", savename("train", parameters, "csv"), train_df)
# wsave(datadir("classifier", savename("val", parameters, "csv"), val_df)
# wsave(datadir("classifier", savename("test", parameters, "csv"), test_df)

# # save the model
# dict = Dict(:model => full_model)
# safesave(datadir("classifier", savename("model", parameters, "bson")), dict)

### another version of saving
# this one might be easier for evaluation
# save the dataframes to csv
wsave(datadir("classifier", savename(parameters), "train.csv"), train_df)
wsave(datadir("classifier", savename(parameters), "val.csv"), train_df)
wsave(datadir("classifier", savename(parameters), "test.csv"), train_df)

# save the model with results
dict = Dict(
    :model => full_model,
    :parameters => parameters,
    :train_acc => accuracy(encode_labels(ytrain, labelnames), ztrain),
    :val_acc => accuracy(encode_labels(yval, labelnames), zval),
    :test_acc => accuracy(encode_labels(ytest, labelnames), ztest)
)
wsave(datadir("classifier", savename(parameters), "model.bson"), dict)
@info "Results saved."