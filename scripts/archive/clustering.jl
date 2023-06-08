using DrWatson
using Mill, Flux, DataFrames
using Statistics
using PrettyTables
using CSV

include(srcdir("dataset.jl"))
include(srcdir("data.jl"))
include(srcdir("constructors.jl"))
include(srcdir("paths.jl"))

modelname = "hmill_classifier_crossentropy"
df = collect_results(expdir("cuckoo_small", model), subfolders=true)
sort!(df, :val_acc, rev=true)

model = df.model[1];

# load data
d = Dataset()
labels = d.family
const labelnames = sort(unique(labels))

# split data
train_ix, val_ix, test_ix = load_split_indexes(split_path)
Xtrain, ytrain = d[train_ix]
Xval, yval = d[val_ix]
Xtest, ytest = d[test_ix]

enc = model[1](Xtrain)


function load_features()
    df = CSV.read()
end


using Clustering, Distances

"""
    calculate_clustering(enc, n)

Provided the features `enc` and number of clusters, calculates all types of clustering
results and returns the cluster assignments.
"""
function calculate_clustering(enc, n)
    # kmeans
    c_means = kmeans(enc, n)
    # randindex(c_means, encode_labels(ytrain, labelnames))

    # distance matrix (needed for other clustering methods)
    M = pairwise(SqEuclidean(), enc)

    # medoids
    c_medoids = kmedoids(M, n)
    # randindex(c_medoids, encode_labels(ytrain, labelnames))

    # hierarchical
    ha = hclust(M, linkage=:average);
    hs = hclust(M, linkage=:single);
    hw = hclust(M, linkage=:ward_presquared);

    la = cutree(ha, k=10);
    # randindex(la, l)
    ls = cutree(hs, k=10);
    # randindex(ls, l)
    lw = cutree(hw, k=10);
    # randindex(lw, l)

    return Int8.(c_means.assignments), Int8.(c_medoids.assignments), Int8.(la), Int8.(ls), Int8.(lw)
end

n = 10 # number of clusters
cmeans, cmedoids, ha, hs, hw = calculate_clustering(enc, n)
modelname = "hmill_classifier_crossentropy"

l = encode_labels(ytrain, labelnames)

clustering_df = DataFrame(
    :ground_truth => l,
    :kmeans => cmeans,
    :kmedoids => cmedoids,
    :hierarchical_average => ha,
    :hierarchical_single => hs,
    :hierarchical_ward => hw
)

safesave(expdir("cuckoo_small", "clustering", modelname, "n=$n.csv"), clustering_df)

function clustering_results(filename::String)
    df = CSV.read(filename, DataFrame)

    r = DataFrame(
        hcat([vcat(randindex(df.ground_truth, df[:, i])...) for i in 2:5]...),
        [
            :adjusted_RI,
            :RI,
            :mirkin_index,
            :hubert_index
        ]
    )
    pretty_table(r, nosubheader=true, formatters = ft_round(4))
end

using UMAP
embedding = umap(enc, 2, n_neighbors=30)

include(srcdir("plotting.jl"))

scatter2(embedding, zcolor=l, markerstrokewidth=0, ms=2, color=:jet)

