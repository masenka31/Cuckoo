using Random
using StatsBase

"""
    random_ix(n::Int, seed=nothing)

This function generates random indexes based on the maximum number
and given seed. If no seed is set, samples randomly.
"""
function random_ix(n::Int, seed=nothing)
    # set seed
    (seed == nothing) ? nothing : Random.seed!(seed)

    _ix = sample(1:n, n; replace=false)

    # reset seed
	(seed !== nothing) ? Random.seed!() : nothing

    return _ix
end

"""
	seqids2bags(bagids)
"""
function seqids2bags(bagids)
	c = countmap(bagids)
	Mill.length2bags([c[i] for i in sort(collect(keys(c)))])
end

"""
	reindex(bagnode, inds)
A faster implementation of Base.getindex.
"""
function reindex(bagnode, inds)
	obs_inds = bagnode.bags[inds]
	new_bagids = vcat(map(x->repeat([x[1]], length(x[2])), enumerate(obs_inds))...)
	data = bagnode.data.data[:,vcat(obs_inds...)]
	new_bags = GroupAD.seqids2bags(new_bagids)
	BagNode(ArrayNode(data), new_bags)
end

function train_test_ix(y::Vector; ratio=0.5, seed=nothing)
    n = length(y)
    n1 = round(Int, ratio*n)

    # get the indexes
    _ix = random_ix(n, seed)
    
    # split indexes to train/test
    train_ix, test_ix = _ix[1:n1], _ix[n1+1:end]
end

function train_val_test_ix(y::Vector; ratios=(0.6,0.2,0.2), seed=nothing)
    n = length(y)
    n1 = floor(Int, ratios[1]*n)
    n2 = floor(Int, ratios[2]*n)

    # get the indexes
    _ix = random_ix(n, seed)
    
    # split indexes to train/test
    train_ix, val_ix, test_ix = _ix[1:n1], _ix[n1+1:n1+n2], _ix[n1+n2+1:end]
end

"""
    train_test_split(d::Dataset, y::Vector{String}; ratio::Number=0.5, seed=nothing)

Classic train/test split with given ratio.
"""
function train_test_split(d::Dataset, y::Vector{String}; ratio::Number=0.5, seed=nothing)
    train_ix, test_ix = train_test_ix(y; ratio=ratio, seed=seed)
    Xtrain, ytrain = d[train_ix]
    Xtest, ytest = d[test_ix]
    return (Xtrain, ytrain), (Xtest, ytest)
end

"""
    train_val_test_split(d::Dataset, y::Vector{String}; ratios::Tuple=(0.6,0.2,0.2), seed=nothing)

Classic train/test split with given ratio.
"""
function train_val_test_split(d::Dataset, y::Vector{String}; ratios::Tuple=(0.6,0.2,0.2), seed=nothing)
    train_ix, val_ix, test_ix = train_val_test_ix(y; ratios=ratios, seed=seed)
    Xtrain, ytrain = d[train_ix]
    Xval, yval = d[val_ix]
    Xtest, ytest = d[test_ix]
    return (Xtrain, ytrain), (Xval, yval), (Xtest, ytest)
end

##############################################################################
### Check the encoding of everything since there is some kind of a mistake!!!
##############################################################################

function encode_labels(y, labelnames=nothing)
    isnothing(labelnames) ? labelnames = sort(unique(y)) : nothing
    categorical_labels = Int8[]
    d = Dict(labelnames .=> 1:length(labelnames))
    for yi in y
        push!(categorical_labels, d[yi])
    end
    return categorical_labels
end

function decode_labels(y::Vector{Int8}, labelnames)
    d = Dict(1:length(labelnames) .=> labelnames)
    string_labels = String[]
    for yi in train_df.ytrain
        push!(string_labels, d[yi])
    end
    return string_labels
end

# function load_split_indexes(path::String)
#     df = CSV.read(joinpath(path, "split_indexes.csv"), DataFrame)
#     train_ix = df.train_ix
#     val_ix = df.val_ix[df.val_ix .!= -1]
#     test_ix = df.test_ix[df.test_ix .!= -1]
#     return train_ix, val_ix, test_ix
# end
