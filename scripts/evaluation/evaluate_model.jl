using DrWatson
using Cuckoo
using Mill, Flux, DataFrames, CSV
using Statistics
using PrettyTables, ProgressMeter

if isempty(ARGS)
    modelname = "hmill_classifier_crossentropy"
    dataset = "cuckoo_small"
else
    modelname = ARGS[1]
    dataset = ARGS[2]
end

# df = collect_results(datadir(model), subfolders=true, rexclude=[r"seed=1", r"activation=sigmoid"])
# g = groupby(df, :parameters)
# c = combine(g, [:train_acc, :val_acc, :test_acc] .=> mean, renamecols=false)
# sort!(c, :val_acc, rev=true)
# p = DataFrame(c.parameters)

# results = hcat(c[:, Not(:parameters)], p)

# pretty_table(first(results, 5), nosubheader=true, crop=:none, formatters = ft_round(3, [1,2,3]))

accuracy(y1::Vector{T}, y2::Vector{T}) where T = mean(y1 .== y2)

folders = readdir(expdir(dataset, modelname), join=true)

results = DataFrame[]
@showprogress "Calculating accuracies..." for f in folders
    tr_df = CSV.read(joinpath(f, "train.csv"), DataFrame)
    val_df = CSV.read(joinpath(f, "val.csv"), DataFrame)
    ts_df = CSV.read(joinpath(f, "test.csv"), DataFrame)

    tr_acc, val_acc, test_acc = [accuracy(d[:,1], d[:,2]) for d in [tr_df, val_df, ts_df]]
    push!(
        results,
        DataFrame(:train_acc => tr_acc, :val_acc => val_acc, :test_acc => test_acc, :foldername => f)# split(f, "/")[end])
    )
end

results = vcat(results...)
sort!(results, :val_acc, rev=true)

pretty_table(first(results, 5), nosubheader=true, crop=:none, formatters = ft_round(3, [1,2,3]))