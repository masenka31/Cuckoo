using DrWatson
using Mill, Flux, DataFrames
using Statistics
using PrettyTables

if isempty(ARGS)
    model = "classifier"
else
    model = ARGS[1]
end

df = collect_results(datadir(model), subfolders=true, rexclude=[r"seed=1", r"activation=sigmoid"])
g = groupby(df, :parameters)
c = combine(g, [:train_acc, :val_acc, :test_acc] .=> mean, renamecols=false)
sort!(c, :val_acc, rev=true)
p = DataFrame(c.parameters)

results = hcat(c[:, Not(:parameters)], p)

pretty_table(first(results, 5), nosubheader=true, crop=:none, formatters = ft_round(3, [1,2,3]))
