using DrWatson
using CSV, DataFrames
using Statistics
using ProgressMeter

if isempty(ARGS)
    model = "classifier"
else
    model = ARGS[1]
end

accuracy(y1::Vector{T}, y2::Vector{T}) where T = mean(y1 .== y2)

parentf = readdir(datadir(model), join=true, sort=false)
results_df = DataFrame[]
@showprogress for f in parentf
    tr = CSV.read(joinpath(f, "train.csv"), DataFrame)
    tr_acc = accuracy(tr[:,1], tr[:,2])
    val = CSV.read(joinpath(f, "val.csv"), DataFrame)
    val_acc = accuracy(val[:,1], val[:,2])
    ts = CSV.read(joinpath(f, "test.csv"), DataFrame)
    ts_acc = accuracy(ts[:,1], ts[:,2])
    d = DataFrame([:train, :val, :test] .=>[tr_acc, val_acc, ts_acc])
    push!(results_df, d)
end
results_df = vcat(results_df...)

function extract_parameters(foldername::String, model::String)
    r = ".*$model/(.*)"
    m = match(Regex(r), foldername)
    name = String(m.captures[1])

    s = split(name, "_")
    d = Dict()
    for par in s
        ss = split(par, "=")
        push!(d, ss[1] => ss[2])
    end
    return d
end
