using DrWatson
using DataFrames, CSV
using PrettyTables

# load results csv file
results_file = ARGS[1]
df = CSV.read(results_file, DataFrame)

gdf = groupby(df, :split)

acc = []
for g in gdf
    y = g.ground_truth
    yhat = g.predicted
    a = mean(y .== yhat)
    push!(acc, [g.split[1], a])
end

acc = vcat(acc...)
accuracy = DataFrame(acc)
pretty_table(accuracy)