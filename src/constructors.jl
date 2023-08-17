using Flux

"""
    classifier_constructor(Xtrain, hdim, activation, aggregation, nlayers; seed = nothing)

Constructs a classifier as a model composed of Mill model and simple Chain of Dense layers.
The output dimension is fixed to be 10, `hdim` is the hidden dimension in both Mill model
the Chain model.
"""
function hmil_classifier_constructor(Xtrain; hdim, activation, aggregation, nlayers, bag_count, dropout_p = 0, odim = 10, seed = nothing, kwargs...)
    
    # set seed
    (seed == nothing) ? nothing : Random.seed!(seed)

    # activation from string to function
    activation = eval(Symbol(activation))

    # add BagCount if provided and aggregation from string to function
    if bag_count
        aggregation = BagCount ∘ eval(Symbol(aggregation))
    else
        aggregation = eval(Symbol(aggregation))
    end

    # create mill model
    m = reflectinmodel(
        Xtrain[1],
        k -> Dense(k, hdim, activation),
        d -> aggregation(d)
    )

    # else create the net after Mill model and add it
    net = dense_classifier_constructor(hdim, odim; hdim=hdim, activation=activation, nlayers=nlayers, dropout_p=dropout_p, kwargs...)
    # connect the full model
    full_model = Chain(m, net)

    # reset seed
	(seed !== nothing) ? Random.seed!() : nothing

    # try that the model works
    try
        full_model(Xtrain[1])
    catch e
        error("Model wrong, error: $e")
    end

    return full_model
end

"""
    dense_classifier_constructor(idim::Int, odim::Int; hdim::Int, activation::Union{Function, String}, nlayers::Int, dropout_p::Number, kwargs...)

Constructs a neural network classifier with dense layers and specified parameters. If `dropout_p == 0`,
does not use any dropout layer, otherwise there is a dropout layer.
"""
function dense_classifier_constructor(
    idim::Int, odim::Int;
    hdim::Int, activation::Union{Function, String}, nlayers::Int, dropout_p::Number, kwargs...
    )
    if typeof(activation) == String
        act = eval(Symbol(activation))
    else
        act = activation
    end

    if dropout_p == 0
        if nlayers == 1
            model = Dense(hdim, odim)
        elseif nlayers == 2
            model = Chain(Dense(hdim, hdim, act), Dense(hdim, odim))
        elseif nlayers == 3
            model = Chain(Dense(hdim, hdim, act), Dense(hdim, hdim, act), Dense(hdim, odim))
        elseif nlayers == 4
            model = Chain(Dense(hdim, hdim, act), Dense(hdim, hdim, act), Dense(hdim, hdim, act), Dense(hdim, odim))
        else
            error("Number of layers is wrong, `nlayers` ∈ [0,1,2,3,4].")
        end
    else
        # add dropout after each dense layer, if nlayers > 1
        if nlayers == 1
            model = Dense(hdim, odim)
        elseif nlayers == 2
            model = Chain(Dense(idim, hdim, act), Dropout(dropout_p), Dense(hdim, odim))
        elseif nlayers == 3
            model = Chain(Dense(idim, hdim, act), Dropout(dropout_p), Dense(hdim, hdim, act), Dropout(dropout_p), Dense(hdim, odim))
        elseif nlayers == 4
            model = Chain(Dense(idim, hdim, act), Dropout(dropout_p), Dense(hdim, hdim, act), Dropout(dropout_p), Dense(hdim, hdim, act), Dropout(dropout_p), Dense(hdim, odim))
        else
            error("Number of layers is wrong, `nlayers` ∈ [0,1,2,3,4].")
        end
    end
    return model
end