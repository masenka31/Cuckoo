using Clustering, Distances

load_features(filename::String) = collect(CSV.read(filename, DataFrame) |> Array |> transpose)

function load_feature_files(modelname::String)
    folders = readdir(expdir("cuckoo_small", modelname), join=true)
    train_files = joinpath.(folders, "train_features.csv")
    val_files = joinpath.(folders, "val_features.csv")
    test_files = joinpath.(folders, "test_features.csv")

    return train_files, val_files, test_files
end

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

"""
    cluster_and_save(modelname::String, enc, n::Int)

Calculates clustering assignments and saves the ground truth labels
as well as the assignments to csv file.
"""
function cluster_and_save(modelname::String, enc, n::Int)
    cmeans, cmedoids, ha, hs, hw = calculate_clustering(enc, n)

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
end

"""
    clustering_results(filename::String)

From a csv file that contains a DataFrame with clustering assignments,
calculates RandIndex and prints output table.
"""
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