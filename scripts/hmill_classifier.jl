using DrWatson
include(srcdir("paths.jl"))
include(srcdir("dataset.jl"))
include(srcdir("data.jl"))
include(srcdir("constructors.jl"))
include(srcdir("utils.jl"))

using Flux
using Flux: @epochs

modelname = ARGS[1]
seed = parse(Int, ARGS[3])
rep = parse(Int, ARGS[4])
labels_file = datadir("labels.csv")
df = CSV.read(labels_file, DataFrame)
tr_ratio = 60

split_df = load_split(seed, tr_ratio)

# load data
d = Dataset()
const labelnames = sort(unique(df.family))

# split data
r = collect(1:length(df.family))
match_df = innerjoin(df, DataFrame(:ix => r, :hash => d.samples), split_df, on=:hash)

train_ix = match_df.ix[match_df.split .== "train"]
val_ix = match_df.ix[match_df.split .== "validation"]
test_ix = match_df.ix[match_df.split .== "test"]

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
    x, y = Xtrain[ix], ytrain[ix]
    yoh = Flux.onehotbatch(y, labelnames)
    return (x, yoh)
end

##########################################
### Parameters, model, loss & accuracy ###
##########################################

function sample_params(seed)
    Random.seed!(seed)

    mdim = sample([8,16,32,64])
    activation = sample(["sigmoid", "tanh", "relu", "swish"])
    aggregation = sample(["SegmentedMeanMax", "SegmentedMax", "SegmentedMean"])
    nlayers = sample(1:3)
    batchsize = sample([32,64,128,256])

    Random.seed!()
    return (mdim=mdim, activation=activation, aggregation=aggregation, nlayers=nlayers,batchsize=batchsize)
end

# Parameters
parameters = sample_params(seed)

# loss and accuracy
loss(X, y) = Flux.logitcrossentropy(full_model(X), y)
accuracy(x::ProductNode, y::Vector{String}) = mean(labelnames[Flux.onecold(full_model(x))] .== y)
accuracy(x::ProductNode, yoh::Flux.OneHotMatrix) = mean(labelnames[Flux.onecold(full_model(x))] .== Flux.onecold(yoh, labelnames))
accuracy(y1::Vector{T}, y2::Vector{T}) where T = mean(y1 .== y2)

# initialize the model
xtr, _ = minibatch(d, train_ix)
full_model = classifier_constructor(xtr; parameters..., odim=length(labelnames));
mill_model = full_model[1];

# initialize optimizer
opt = ADAM()

@info "Model created, starting training..."

# train the model
max_train_time = 60*15 # 20 minutes of training time

start_time = time()
while time() - start_time < max_train_time
    # batches = map(_ -> minibatch(d, train_ix, batchsize=parameters.batchsize), 1:10);
    batches = map(_ -> minibatch(Xtrain, ytrain, batchsize=parameters.batchsize), 1:10);
    Flux.train!(loss, Flux.params(full_model), batches, opt)
end

### SAVING

using UUIDs
id = "$(uuid1())"

par_dict = Dict(keys(parameters) .=> values(parameters))
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
s_tr = full_model(Xtrain)
s_val = full_model(Xval)
s_test = full_model(Xtest)

ztrain = encode_labels(Flux.onecold(s_tr, labelnames), labelnames)
zval = encode_labels(Flux.onecold(s_val, labelnames), labelnames)
ztest = encode_labels(Flux.onecold(s_test, labelnames), labelnames)

predictions = vcat(ztrain, zval, ztest)

softmax_output = DataFrame(
    hcat(
        s_tr,
        s_val,
        s_test
    ) |> transpose |> collect, :auto)


# add softmax output
results_df = DataFrame(
    :hash => vcat(
        match_df.hash[match_df.split .== "train"],
        match_df.hash[match_df.split .== "validation"],
        match_df.hash[match_df.split .== "test"],
    ),
    :ground_truth => vcat(
        encode_labels(ytrain, labelnames),
        encode_labels(yval, labelnames),
        encode_labels(ytest, labelnames)
    ),
    :predicted => predictions,
    :split => vcat(
        repeat(["train"], length(ytrain)),
        repeat(["validation"], length(yval)),
        repeat(["test"], length(ytest))
    )
)

final_df = hcat(results_df, softmax_output)

@info "Saving results"
safesave(expdir("cuckoo_small", modelname, "dense_classifier", "$id.bson"), results_dict)
safesave(expdir("cuckoo_small", modelname, "dense_classifier", "$id.csv"), results_df)
@info "Results saved, experiment finished."