using DrWatson
include(srcdir("dataset.jl"))
include(srcdir("data.jl"))
include(srcdir("constructors.jl"))

using Flux
using Flux: @epochs

# load data and labels
d = Dataset()
labels = d.family
const labelnames = sort(unique(labels))
# data, _ = d[:]

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

# function to sample parameters
function sample_params()
    mdim = sample([8,16,32,64])
    activation = sample(["tanh", "relu", "swish"])
    aggregation = sample(["SegmentedMeanMax", "SegmentedMax", "SegmentedMean"])
    nlayers = sample(1:3)
    return (mdim=mdim, activation=activation, aggregation=aggregation, nlayers=nlayers)
end

# sample parameters
const parameters = sample_params()
max_seed = parse(Int, ARGS[1])

for seed in 1:max_seed
    
    # loss and accuracy
    loss(X, y) = Flux.logitcrossentropy(full_model(X), y)
    accuracy(x::ProductNode, y::Vector{String}) = mean(labelnames[Flux.onecold(full_model(x))] .== y)
    accuracy(x::ProductNode, yoh::Flux.OneHotMatrix) = mean(labelnames[Flux.onecold(full_model(x))] .== Flux.onecold(yoh, labelnames))
    accuracy(y1::Vector{T}, y2::Vector{T}) where T = mean(y1 .== y2)

    # split data
    ratios=(0.2,0.4,0.4)
    train_ix, val_ix, test_ix = train_val_test_ix(labels; ratios=ratios, seed=seed)
    Xtrain, ytrain = d[train_ix]
    Xval, yval = d[val_ix]
    Xtest, ytest = d[test_ix]

    # create the model
    xtr, _ = minibatch(d, train_ix)
    full_model = classifier_constructor(xtr; parameters..., odim=length(labelnames));
    mill_model = full_model[1];

    # initialize optimizer
    opt = ADAM()

    @info "Everything set, starting training..."

    # train the model for 10 minutes
    start_time = time()
    max_train_time = 60#*10 # 10 minutes of training time
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

    # save the dataframes to csv
    wsave(datadir("classifier", savename("seed=$seed", parameters), "train.csv"), train_df)
    wsave(datadir("classifier", savename("seed=$seed", parameters), "val.csv"), val_df)
    wsave(datadir("classifier", savename("seed=$seed", parameters), "test.csv"), test_df)

    # save the model with results
    dict = Dict(
        :model => full_model,
        :parameters => parameters,
        :seed => seed,
        :train_acc => accuracy(encode_labels(ytrain, labelnames), ztrain),
        :val_acc => accuracy(encode_labels(yval, labelnames), zval),
        :test_acc => accuracy(encode_labels(ytest, labelnames), ztest)
    )
    wsave(datadir("classifier", savename("seed=$seed", parameters), "results.bson"), dict)
    @info "Results for seed=$seed saved."
end

@info "Experiment finished."