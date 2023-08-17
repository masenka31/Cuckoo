START = time()

"""
Note: Expect at least 60 minutes of training for each repetition.
20G, how many minutes? give it more, validation early stopping working, hopefully
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
if isempty(ARGS)
    seed = 1
    rep_start = 1
    rep_end = 10
else
    seed = parse(Int, ARGS[1])
    rep_start = parse(Int, ARGS[2])
    rep_end = parse(Int, ARGS[3])
end
ratio = "timesplit"
dataset = "cuckoo_small"

# load data and split into train/validation/test
d = Dataset(dataset)
labels = d.family
const labelnames = sort(unique(labels))

@time tr_x, tr_l, tr_h, val_x, val_l, val_h, test_x, test_l, test_h = load_split_indexes(
    d, seed=seed, ratio=ratio
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

    hdim = sample([8,16,32,64,128,256])
    activation = sample(["sigmoid", "tanh", "relu", "swish"])
    nlayers = sample(1:4)
    batchsize = sample([32,64,128,256])
    aggregation = sample(["SegmentedMeanMax", "SegmentedMax", "SegmentedMean"])
    dropout_p = sample([0, 0.1, 0.2, 0.3])
    bag_count = sample([true, false])

    Random.seed!()
    return (hdim=hdim, activation=activation, nlayers=nlayers, batchsize=batchsize, aggregation=aggregation, dropout_p=dropout_p, bag_count=bag_count)
end

for rep in rep_start:rep_end
    parameters = sample_params(rep)

    # loss and accuracy
    loss(X, y) = Flux.logitcrossentropy(full_model(X), y)
    accuracy(x::ProductNode, y::Vector{String}) = mean(labelnames[Flux.onecold(full_model(x))] .== y)
    accuracy(x::ProductNode, yoh::Flux.OneHotMatrix) = mean(labelnames[Flux.onecold(full_model(x))] .== Flux.onecold(yoh, labelnames))
    accuracy(y1::Vector{T}, y2::Vector{T}) where T = mean(y1 .== y2)

    # create the model
    xtr, ytr = minibatch(tr_x, tr_l, batchsize=2)
    full_model = hmil_classifier_constructor(xtr; parameters..., odim=10);
    mill_model = full_model[1];

    # initialize optimizer
    opt = ADAM(1e-4)

    @info "Everything set, starting training..."

    # train the model
    Flux.trainmode!(full_model)
    Flux.train!(loss, Flux.params(full_model), [(xtr, ytr)], opt) # initialize the training function
    max_train_time = 60*60 # 1 hour of training
    
    check_ix = 0
    best_val_loss = Inf
    best_model = deepcopy(full_model)
    start_time = time()
    while time() - start_time < max_train_time
        batches = map(_ -> minibatch(tr_x, tr_l), 1:5);
        Flux.train!(loss, Flux.params(full_model), batches, opt)
        if check_ix % 10 == 0
            # every 10 passes of the training, calculate full validation loss
            val_loss = loss(val_x, Flux.onehotbatch(val_l, labelnames))
            @info "val_loss = $val_loss"
            if val_loss < best_val_loss
                # if validation loss gets better, save the best model
                best_val_loss = val_loss
                best_model = deepcopy(full_model)
            end
        else
            # in other cases, just calculate batch loss
            # batch_loss = mean(x -> loss(x...), batches)
            # @info "batch loss = $(round(batch_loss, digits=3))"
        end
        check_ix += 1
    end

    # get only the best model of all
    full_model = deepcopy(best_model)

    ######################
    ### Saving results ###
    ######################

    Flux.testmode!(full_model)

    id = "$(uuid1())"   # generate unique uuid to save data in two files - one contains metadata, the other contains the predictions

    # parameters dictionary
    par_dict = Dict(keys(parameters) .=> values(parameters))
    info_dict = Dict(
        :uuid => id,
        :feature_model => modelname,
        :nn_model => "dense_classifier",
        :seed => seed,
        :repetition => rep,
        :ratio => ratio,
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
    # @time softmax_output = DataFrame(
    #     hcat(
    #         hcat(map(i -> softmax(full_model(tr_x[i])), 1:nobs(tr_x))...),
    #         hcat(map(i -> softmax(full_model(val_x[i])), 1:nobs(val_x))...),
    #         hcat(map(i -> softmax(full_model(test_x[i])), 1:nobs(test_x))...)
    #     ) |> transpose |> collect, :auto
    # )

    # final_df = hcat(results_df, softmax_output)
    @info "Results calculated."

    df = cuckoo_hash_to_int(results_df)

    @info "Saving results..."
    safesave(expdir("results", dataset, modelname, "dense_classifier", "$id.bson"), results_dict)
    safesave(expdir("results", dataset, modelname, "dense_classifier", "$id.csv"), df)
    @info "Results saved, experiment no. $rep finished."

end

@info "Experiment finished in $(time() - START) seconds."