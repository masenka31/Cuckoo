START = time()

"""
Note: Expect at least 30 minutes of training for each repetition.
"""

using DrWatson
using Cuckoo
using JsonGrinder
using Mill
using StatsBase
using DataFrames
using Flux
using UUIDs

# get the passed arguments
modelname = "hmil"
seed = parse(Int, ARGS[1])
rep_start = parse(Int, ARGS[2])
rep_end = parse(Int, ARGS[3])
tr_ratio = "timesplit"

# load data and split into train/validation/test
d = Dataset()
labels = d.family
const labelnames = sort(unique(labels))

@time tr_x, tr_l, tr_h, val_x, val_l, val_h, test_x, test_l, test_h = load_split_indexes(
    d, seed=seed, tr_ratio=tr_ratio
);

function minibatch(X::ProductNode, y; batchsize=64)
    ix = sample(1:length(y), batchsize)
    labelnames = sort(unique(y))
    x, y = X[ix], y[ix]
    yoh = Flux.onehotbatch(y, labelnames)
    return (x, yoh)
end

using Random
# fix seed to always choose the same hyperparameters
function sample_params(seed)
    Random.seed!(seed)

    mdim = sample([8,16,32,64,128,256])
    activation = sample(["sigmoid", "tanh", "relu", "swish"])
    nlayers = sample(2:4)
    batchsize = sample([32,64,128,256])
    aggregation = sample(["SegmentedMeanMax", "SegmentedMax", "SegmentedMean"])

    Random.seed!()
    return (mdim=mdim, activation=activation, nlayers=nlayers, batchsize=batchsize, aggregation=aggregation)
end

@time for rep in rep_start:rep_end
    parameters = sample_params(rep)

    # loss and accuracy
    loss(X, y) = Flux.logitcrossentropy(full_model(X), y)
    accuracy(x::ProductNode, y::Vector{String}) = mean(labelnames[Flux.onecold(full_model(x))] .== y)
    accuracy(x::ProductNode, yoh::Flux.OneHotMatrix) = mean(labelnames[Flux.onecold(full_model(x))] .== Flux.onecold(yoh, labelnames))
    accuracy(y1::Vector{T}, y2::Vector{T}) where T = mean(y1 .== y2)

    # create the model
    xtr, ytr = minibatch(tr_x, tr_l, batchsize=2)
    full_model = classifier_constructor(xtr; parameters..., odim=10);
    mill_model = full_model[1];

    # initialize optimizer
    opt = ADAM(1e-4)

    @info "Everything set, starting training..."

    # train the model for 15 minutes
    Flux.train!(loss, Flux.params(full_model), [(xtr, ytr)], opt)
    max_train_time = 60*30 # 15 minutes of training time
    
    start_time = time()
    while time() - start_time < max_train_time
        batches = map(_ -> minibatch(tr_x, tr_l), 1:5);
        Flux.train!(loss, Flux.params(full_model), batches, opt)
        batch_loss = mean(x -> loss(x...), batches)
        @info "batch loss = $(round(batch_loss, digits=3))"
    end

    ######################
    ### Saving results ###
    ######################

    id = "$(uuid1())"   # generate unique uuid to save data in two files - one contains metadata, the other contains the predictions

    # parameters dictionary
    par_dict = Dict(keys(parameters) .=> values(parameters))
    info_dict = Dict(
        :uuid => id,
        :feature_model => modelname,
        :nn_model => "dense_classifier",
        :seed => seed,
        :repetition => rep,
        :tr_ratio => tr_ratio,
        :model => full_model
    )

    results_dict = merge(par_dict, info_dict)

    @time predictions = vcat(
        encode_labels(vcat(map(i -> Flux.onecold(full_model(tr_x[i]), labelnames), 1:nobs(tr_x))...), labelnames),
        encode_labels(vcat(map(i -> Flux.onecold(full_model(val_x[i]), labelnames), 1:nobs(val_x))...), labelnames),
        encode_labels(vcat(map(i -> Flux.onecold(full_model(test_x[i]), labelnames), 1:nobs(test_x))...), labelnames)
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
        :split => vcat(
            repeat(["train"], length(tr_h)),
            repeat(["validation"], length(val_h)),
            repeat(["test"], length(test_h))
        )
    )

    # add the softmax output
    @time softmax_output = DataFrame(
        hcat(
            hcat(map(i -> softmax(full_model(tr_x[i])), 1:nobs(tr_x))...),
            hcat(map(i -> softmax(full_model(val_x[i])), 1:nobs(val_x))...),
            hcat(map(i -> softmax(full_model(test_x[i])), 1:nobs(test_x))...)
        ) |> transpose |> collect, :auto
    )

    final_df = hcat(results_df, softmax_output)
    @info "Results calculated."

    @info "Saving results..."
    safesave(expdir("results", modelname, "dense_classifier", "timesplit", "$id.bson"), results_dict)
    safesave(expdir("results", modelname, "dense_classifier", "timesplit", "$id.csv"), results_df)
    @info "Results saved, experiment no. $rep finished."

end

@info "Experiment finished in $(time() - START) seconds."