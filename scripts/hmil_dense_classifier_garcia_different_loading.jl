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
using ProgressMeter
using JLD2

function garcia_index_mapping(indexes, bit_index)
    full_index = collect(1:19997)
    index_mask = map(x -> in(x, indexes), full_index)

    missing_index = Int[]
    k = 1
    for i in 1:19997
        if bit_index[i]
            push!(missing_index, k)
            k += 1
        else
            push!(missing_index, 0)
        end
    end

    # sort(countmap(missing_index[index_mask]))
    return filter(x -> x != 0, missing_index[index_mask]), missing_index[index_mask] .!= 0
end

# get the passed arguments
modelname = "hmil"
if length(ARGS) == 0
    seed = 1
    rep_start = 1
    rep_end = 1
    rep = 1
else
    seed = parse(Int, ARGS[1])
    rep_start = parse(Int, ARGS[2])
    rep_end = parse(Int, ARGS[3])
end
ratio = "garcia"

# load data and split into train/validation/test
d = Dataset("garcia")
labels = d.family
const labelnames = sort(unique(labels))

# load only indexes of the data
_train_ix, _val_ix, _test_ix, train_h, val_h, test_h = load_indexes(d, seed=seed)

@info "Indexes loaded."

# load saved data
dt = load_object(projectdir("garcia_productnodes.jld2"))
data, labels, bit_index = dt

# prepare the indexes without the unparsed samples
train_ix, train_mask = garcia_index_mapping(_train_ix, bit_index)
train_h = train_h[train_mask]
val_ix, val_mask = garcia_index_mapping(_val_ix, bit_index)
val_h = val_h[val_mask]
test_ix, test_mask = garcia_index_mapping(_test_ix, bit_index)
test_h = test_h[test_mask]

@info "Indexes mapped."

# get training, validation and test data
# @time test_data = catobs([data[ix] for ix in test_ix]);
test_l = (labels[test_ix] .== labelnames[2])';

@info "Test data prepared."

# @time train_data = catobs([data[ix] for ix in train_ix]);
train_l = (labels[train_ix] .== labelnames[2])';

@info "Train data prepared."

# @time val_data = catobs([data[ix] for ix in val_ix]);
val_l = (labels[val_ix] .== labelnames[2])';

@info "Validation data prepared."

function minibatch(X::ProductNode, y; batchsize=64)
    ix = sample(1:length(y), batchsize)
    labelnames = sort(unique(y))
    x, y = X[ix], y[ix]
    ybool = y .== labelnames[2]
    return (x, ybool')
end
function minibatch(data::ProductNode, labels, indexes; batchsize=64)
    ixs = sample(indexes, batchsize)
    labelnames = sort(unique(labels))
    x = catobs([data[ix] for ix in ixs])
    y = labels[ixs]
    ybool = y .== labelnames[2]
    return (x, ybool')
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

@time for rep in rep_start:rep_end
    parameters = sample_params(rep)

    # loss and accuracy
    loss(X, y) = Flux.logitbinarycrossentropy(full_model(X), y)
    accuracy(y1::Vector{T}, y2::Vector{T}) where T = mean(y1 .== y2)

    # create the model
    # xtr, ytr = minibatch(d, train_ix, batchsize=2)
    # xtr, ytr = minibatch(train_data, train_l, batchsize=parameters.batchsize)
    xtr, ytr = minibatch(data, labels, train_ix, batchsize=parameters.batchsize)
    full_model = hmil_classifier_constructor(data; parameters..., odim=1);
    mill_model = full_model[1];

    # initialize optimizer
    opt = ADAM(1e-4)

    @info "Everything set, starting training..."

    # train the model for 15 minutes
    Flux.trainmode!(full_model);
    Flux.train!(loss, Flux.params(full_model), [(xtr, ytr)], opt)
    max_train_time = 120#*120 # 2 hours of training time
    
    check_ix = 0
    best_val_loss = Inf
    best_model = deepcopy(full_model)
    start_time = time()

    val_batches = map(_ -> minibatch(data, labels, val_ix, batchsize=parameters.batchsize), 1:10);

    while time() - start_time < max_train_time
        # batches = map(_ -> minibatch(tr_x, tr_l), 1:5);
        batches = map(_ -> minibatch(data, labels, train_ix, batchsize=parameters.batchsize), 1:5);
        Flux.train!(loss, Flux.params(full_model), batches, opt)
        # batch_loss = mean(x -> loss(x...), batches)
        # @info "batch loss = $(round(batch_loss, digits=3))"

        # val_loss = loss(val_data, val_l)
        val_loss = mean(batch -> loss(batch...), val_batches)
        @info "val_loss = $val_loss"
        if val_loss < best_val_loss
            # if validation loss gets better, save the best model
            best_val_loss = val_loss
            best_model = deepcopy(full_model)
        end
    end

    # get only the best model of all
    full_model = deepcopy(best_model)

    ######################
    ### Saving results ###
    ######################

    Flux.testmode!(full_model)

    id = "$(uuid1())"   # generate unique uuid to save data in two files - one contains metadata, the other contains the predictions

    par_dict = Dict(keys(parameters) .=> values(parameters))
    info_dict = Dict(
        :uuid => id,
        :feature_model => modelname,
        :nn_model => "dense_classifier",
        :seed => seed,
        :repetition => rep,
        :model => full_model
    )

    results_dict = merge(par_dict, info_dict)

    # predictions of the model
    # @time tr_pred_labels = full_model(train_data) .> 0.5
    # @time val_pred_labels = full_model(val_data) .> 0.5
    # @time ts_pred_labels = full_model(test_data) .> 0.5
    
    tr_pred_labels = zeros(Float32, length(train_ix))
    for (i, ix) in enumerate(train_ix)
        x = data[ix]
        prob = sigmoid(full_model(x)[1])
        tr_pred_labels[i] = prob > 0.5
    end

    val_pred_labels = zeros(Float32, length(val_ix))
    for (i, ix) in enumerate(val_ix)
        x = data[ix]
        prob = sigmoid(full_model(x)[1])
        val_pred_labels[i] = prob > 0.5
    end

    ts_pred_labels = zeros(Float32, length(test_ix))
    for (i, ix) in enumerate(test_ix)
        x = data[ix]
        prob = sigmoid(full_model(x)[1])
        ts_pred_labels[i] = prob > 0.5
    end

    results_df = DataFrame(
        :hash => vcat(train_h, val_h, test_h),
        :ground_truth => vcat(
            encode_labels(labels[train_ix], labelnames),
            encode_labels(labels[val_ix], labelnames),
            encode_labels(labels[test_ix], labelnames)
        ),
        :predicted => vcat(
            tr_pred_labels .+ 1,
            val_pred_labels .+ 1,
            ts_pred_labels .+ 1
        )[:],
        :split => vcat(
            repeat([0], length(train_h)),
            repeat([1], length(val_h)),
            repeat([2], length(test_h))
        )
    )

    # add the softmax output (for this data, it is not really softmax, but just probability)
    # the model probability is the probability of being class 1 (malicious)
    # softmax_output = DataFrame(
    #     hcat(
    #         1 .- vcat(tr_pred_probabilities, val_pred_probabilities, ts_pred_probabilities), # probability of class 0 (benign)
    #         vcat(tr_pred_probabilities, val_pred_probabilities, ts_pred_probabilities),      # probability of class 1 (malicious)
    #     ), :auto
    # )

    # final_df = hcat(results_df, softmax_output)
    @info "Results calculated."

    @info "Saving results..."
    safesave(expdir("results","garcia", modelname, "dense_classifier", "$id.bson"), results_dict)
    safesave(expdir("results", "garcia", modelname, "dense_classifier", "$id.csv"), results_df)
    @info "Results saved, experiment no. $rep finished."

end

@info "Experiment finished in $(time() - START) seconds."
