using DrWatson
using JsonGrinder
using JSON
using Mill
using ThreadTools
using BSON
using CSV, DataFrames
using ProgressMeter

export read_json

read_json(file) = JSON.parse(read(file, String))

# dataset structure
struct Dataset
    samples
    family
    type
    date
    # schema
    extractor
    name
end

function Dataset(data::String="cuckoo"; full=false)
    if data == "cuckoo"
        if full
            df = CSV.read("$cuckoo_full_path/public_labels.csv", DataFrame)
            sch = BSON.load(datadir("schema_full.bson"))[:schema]
            samples = Vector(df.sha256)
            files = joinpath.(cuckoo_full_path, "public_small_reports", string.(samples, ".json"))
        else
            df = CSV.read("$cuckoo_path/public_labels.csv", DataFrame)
            sch = BSON.load(datadir("schema.bson"))[:schema]
            samples = Vector(df.sha256)
            files = joinpath.(cuckoo_path, "public_small_reports", string.(samples, ".json"))
        end
        extractor = suggestextractor(sch)
        
    elseif data == "garcia"
        # for garcia data, for now, load the original small cuckoo schema
        # and get the extractor from that
        df = CSV.read("/mnt/data/jsonlearning/garcia/reports/labels.csv", DataFrame)
        files = df.samples
        # sch = BSON.load(datadir("schema.bson"))[:schema]
        sch = []
        extractor = BSON.load(datadir("garcia_extractor.bson"))[:extractor]
    end
    
    family = String.(df.classification_family)
    type = String.(df.classification_type)
    date = Vector(df.date)

    Dataset(
        files,
        family,
        type,
        date,
        # schema,
        extractor,
        data
    )
end

function Base.getindex(d::Dataset, inds)
    data = load_samples(d; inds=inds)
    return indexinto(d, data, inds)
end

function indexinto(d::Dataset, data::AbstractMillNode, inds)
    family = d.family[inds]
    return data, family
end
function indexinto(d::Dataset, data::Tuple, inds)
    family = d.family[inds]
    data, mask = data
    return data, family[mask], mask
end
indexinto(d::Dataset, data::Nothing, inds) = nothing


"""
    load_samples(d::Dataset; inds = :)

Reads the samples in Dataset chosen with `inds`. `inds` can be an integer (loads one sample), 
range (loads multiple samples), or `Colon` (loads all samples).

The function is multi-threaded if more than 128 samples are to be loaded. Returns
ProductNode with the number of samples in it.
"""
function load_samples(d::Dataset; inds = :)
    files = d.samples[inds]
    n = length(files)
    dicts = Vector{Union{ProductNode, Missing}}(missing, n)

    if typeof(inds) == Colon || length(inds) > 128
        p = Progress(n; desc="Extracting JSONs: ")
        Threads.@threads for i in 1:n
            # in case the loading errors, returns only the samples it does not error on
            try
                dicts[i] = d.extractor(read_json(files[i]))
            catch e
                @info "Error in filepath = $(files[i])"
                @warn e
            end
            next!(p)
        end
    else
        for i in 1:n
            try
                dicts[i] = d.extractor(read_json(files[i]))
            catch e
                @info "Error in filepath = $(files[i])"
                @warn e
            end
        end
    end

    if sum(ismissing.(dicts)) == n
        return nothing
    elseif sum(ismissing.(dicts)) > 0
        mask = .!ismissing.(dicts)
        return reduce(catobs, dicts[mask |> BitVector]), mask
    else
        return reduce(catobs, dicts)
    end
end


# example usage
# d = Dataset()
# @time d[1:20000]