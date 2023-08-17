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

# get the passed arguments
modelname = "hmil"
if length(ARGS) == 0
    seed = 1
    rep_start = 1
    rep_end = 1
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

# load only indexes of the data (loading all samples might take too much time)
train_ix, val_ix, test_ix, train_h, val_h, test_h = load_indexes(d, seed=seed)

function minibatch(d::Dataset, train_ix; batchsize=64, labelnames=labelnames)
    ix = sample(train_ix, batchsize)
    tmp = d[ix]
    if length(tmp) == 2
        x, y = tmp
    elseif length(tmp) == 3
        x, y, m = tmp
    end
    # ybool = y .== labelnames[2] # 1 if label is malicious
    yoh = Flux.onehotbatch(y, labelnames)
    return (x, yoh)
end

function minibatch(X::ProductNode, y; batchsize=64, labelnames=labelnames)
    ix = sample(1:length(y), batchsize)
    x, y = X[ix], y[ix]
    yoh = Flux.onehotbatch(y, labelnames)
    # ybool = y .== labelnames[2]
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

@time for rep in rep_start:rep_end
    parameters = sample_params(rep)

    # loss and accuracy
    loss(X, y) = Flux.logitcrossentropy(full_model(X), y)
    accuracy(y1::Vector{T}, y2::Vector{T}) where T = mean(y1 .== y2)

    # create the model
    xtr, ytr = minibatch(d, train_ix, batchsize=2)
    full_model = hmil_classifier_constructor(xtr; parameters..., odim=2);
    mill_model = full_model[1];

    # initialize optimizer
    opt = ADAM(1e-4)

    @info "Everything set, starting training..."

    # train the model for 15 minutes
    Flux.trainmode!(full_model);
    Flux.train!(loss, Flux.params(full_model), [(xtr, ytr)], opt)
    max_train_time = 60*120 # 2 hours of training time
    
    check_ix = 0
    best_val_loss = Inf
    best_model = deepcopy(full_model);
    @time val_batches = map(_ -> minibatch(d, val_ix, batchsize=parameters.batchsize), 1:20);
    start_time = time()

    while time() - start_time < max_train_time
        # batches = map(_ -> minibatch(tr_x, tr_l), 1:5);
        batches = map(_ -> minibatch(d, train_ix, batchsize=parameters.batchsize), 1:5);
        Flux.train!(loss, Flux.params(full_model), batches, opt)
        # batch_loss = mean(x -> loss(x...), batches)
        
        @time val_batch_loss = mean(x -> loss(x...), val_batches)
        @info "batch loss = $(round(val_batch_loss, digits=3))"

        if val_batch_loss < best_val_loss
            # if validation loss gets better, save the best model
            best_val_loss = val_batch_loss
            best_model = deepcopy(full_model);
        end
    end

    # get only the best model of all
    full_model = deepcopy(best_model);

    ######################
    ### Saving results ###
    ######################

    Flux.testmode!(full_model);

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

    function predict_labels(indexes)
        # preallocate the vectors
        n = length(indexes)
        labelvec = zeros(Union{Int8, Missing}, n) # label predictions
        probabilities = zeros(Union{Float32, Missing}, n)

        p = Progress(n; desc="Calculating labels and probabilities:")
        Threads.@threads for i in 1:n
            ix = indexes[i]
            tmp = d[ix:ix]
            if isnothing(tmp)
                labelvec[i] = missing
                probabilities[i] = missing
            elseif typeof(tmp[1]) <: ProductNode
                pn = tmp[1]
                proba = softmax(full_model(pn))[2]
                probabilities[i] = proba            # get the probability distribution
                labelvec[i] = round(Int8, proba)    # get the predicted label based on threshold 0.5
            else
                @error "Something went wrong."
            end
            next!(p)
        end
        return labelvec, probabilities
    end

    @time tr_pred_labels, tr_pred_probabilities = predict_labels(train_ix)
    @time val_pred_labels, val_pred_probabilities = predict_labels(val_ix)
    @time ts_pred_labels, ts_pred_probabilities = predict_labels(test_ix)

    # predictions of the model
    results_df = DataFrame(
        :hash => vcat(train_h, val_h, test_h),
        :ground_truth => vcat(
            encode_labels(d.family[train_ix], labelnames),
            encode_labels(d.family[val_ix], labelnames),
            encode_labels(d.family[test_ix], labelnames)
        ),
        :predicted => vcat(
            tr_pred_labels .+ 1,
            val_pred_labels .+ 1,
            ts_pred_labels .+ 1
        ),
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
    safesave(expdir("results", "garcia", modelname * "_ts", "dense_classifier", "$id.bson"), results_dict)
    safesave(expdir("results", "garcia", modelname * "_ts", "dense_classifier", "$id.csv"), results_df)
    @info "Results saved, experiment no. $rep finished."

end

@info "Experiment finished in $(time() - START) seconds."
