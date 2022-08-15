using DrWatson
using Mill, Flux, DataFrames
using Statistics
using PrettyTables

include(srcdir("dataset.jl"))
include(srcdir("data.jl"))
include(srcdir("constructors.jl"))
include(srcdir("paths.jl"))

if isempty(ARGS)
    model = "hmill_classifier_crossentropy"
else
    model = ARGS[1]
end

# load all results, combine and sort by validation accuracy
df = collect_results(expdir("cuckoo_small", model), subfolders=true)#, rexclude=[r"seed=1", r"activation=sigmoid"])
g = groupby(df, :parameters)
c = combine(g, [:train_acc, :val_acc, :test_acc] .=> mean, renamecols=false)
sort!(c, :val_acc, rev=true)
p = DataFrame(c.parameters)
results = hcat(c[:, Not(:parameters)], p)

# find the best models
best_parameters = c.parameters[1]
best_ix = findall(x -> x.parameters == best_parameters, eachrow(df))
best_models = df[best_ix, [:model, :seed]]

# load data and labels
d = Dataset()
labels = d.family
const labelnames = sort(unique(labels))

# choose seed and split data, load them for the model
seed = 2
ratios = (0.2,0.4,0.4)
train_ix, val_ix, test_ix = train_val_test_ix(labels; ratios=ratios, seed=seed)
Xtrain, ytrain = d[train_ix]
Xval, yval = d[val_ix]
Xtest, ytest = d[test_ix]

# load the model itself (make sure that it's the right model for the seed)
model = best_models[1,:model];

# let's see for some plotting functions
using Plots
ENV["GKSwstype"] = "100"
gr(markerstrokewidth=0, ms=2, color=:jet)
using UMAP

# get the encoding of train data from the mill part of the model
# and look at the various dimensions
# and the UMAP representations
enc = model[1](Xtrain)

l = encode_labels(ytrain, labelnames)
scatter2(enc, 2, 5, zcolor=l)
scatter3(enc, rand(1:64, 3)..., zcolor=l)

emb = umap(enc, 3, n_neighbors=30, min_dist=0.2)
scatter3(emb, zcolor=l)

### now to the clustering
using Clustering
using Distances

# kmeans
kmn10 = kmeans(enc, 10);
randindex(kmn10, l)

# kmedoids
M = pairwise(SqEuclidean(), enc)
kmd10 = kmedoids(M, 10);
randindex(kmd10, l)

# hierarchical
ha = hclust(M, linkage=:average);
hs = hclust(M, linkage=:single);
hw = hclust(M, linkage=:ward_presquared);

la = cutree(ha, k=10);
randindex(la, l)
ls = cutree(hs, k=10);
randindex(ls, l)
lw = cutree(hw, k=10);
randindex(lw, l)

### extraction of features
# and saving them

# load data
d = Dataset()
labels = d.family
const labelnames = sort(unique(labels))

# split data
train_ix, val_ix, test_ix = load_split_indexes(split_path)
Xtrain, ytrain = d[train_ix]
Xval, yval = d[val_ix]
Xtest, ytest = d[test_ix]

m = df.model;
p = df.parameters

using ProgressMeter
@showprogress for (model, pars) in zip(m, p)
    enc_train = model[1](Xtrain)
    enc_val = model[1](Xval)
    enc_test = model[1](Xtest)

    dtrain = DataFrame(enc_train', :auto)
    dval = DataFrame(enc_val', :auto)
    dtest = DataFrame(enc_test', :auto)

    safesave(expdir("cuckoo_small", "hmill_classifier_crossentropy", savename(pars), "train_features.csv"), dtrain)
    safesave(expdir("cuckoo_small", "hmill_classifier_crossentropy", savename(pars), "val_features.csv"), dval)
    safesave(expdir("cuckoo_small", "hmill_classifier_crossentropy", savename(pars), "test_features.csv"), dtest)
end
