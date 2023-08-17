using DrWatson
using Cuckoo
using JSON, BSON, JsonGrinder, Mill
using Cuckoo: read_json

using Flux
using UUIDs

###########################################################################################
###                          Schema preparation and extraction                          ###
###########################################################################################

d = Dataset("garcia")

using JsonGrinder, Flux, Mill, MLDataPattern, JSON, HierarchicalUtils, StatsBase, OrderedCollections
using JsonGrinder: DictEntry, Entry

sch = BSON.load(datadir("garcia_schema_run.bson"))[:schema]
extractor = suggestextractor(sch, (; key_as_field=20))

using JsonGrinder, Flux, Mill, MLDataPattern, JSON, HierarchicalUtils, StatsBase, OrderedCollections
using JsonGrinder: DictEntry, Entry

ks_cuckoo = (
    :signatures, :buffer, :metadata, :network, :dropped,
    :info, :extracted, :debug, :virustotal, :target, :strings,
)

extract_data = ExtractDict(deepcopy(extractor.dict))
for k in ks_cuckoo
    delete!(extract_data.dict, k)
end

extract_data[:behavior].dict
delete!(extract_data[:behavior].dict, :processtree)
delete!(extract_data[:behavior].dict, :processes)
delete!(extract_data[:behavior].dict, :apistats)

extract_data[:behavior][:generic]

# delete!(extract_data[:behavior].dict, :generic)

extract_data[:static].dict
delete!(extract_data[:static].dict, :keys)
delete!(extract_data[:static].dict, :imported_dll_count)
delete!(extract_data[:static].dict, :office)
delete!(extract_data[:static].dict, :pdf)
delete!(extract_data[:static].dict, :pdb_path)
delete!(extract_data[:static].dict, :peid_signatures)

### Delete timestamp, since it is the biggest indicator!
# delete!(extract_data[:static].dict, :pe_timestamp)
safesave(datadir("garcia_extractor_wo_timestamp.bson"), Dict(:extractor => extract_data))

# save the prepared extractor
safesave(datadir("garcia_extractor.bson"), Dict(:extractor => extract_data))


using CSV, DataFrames
d = CSV.read("/mnt/data/jsonlearning/garcia/reports/labels.csv", DataFrame)

@time r = read_json(d.samples[1])
@time pn = extract_data(r)

inds = 1:64
files = d.samples[inds]
n = length(files)
dicts = Vector{ProductNode}(undef, n)

using ProgressMeter
t = time()
if typeof(inds) == Colon || length(inds) > 5
    p = Progress(n; desc="Extracting JSONs: ")
    Threads.@threads for i in 1:n
        dicts[i] = extract_data(read_json(files[i]))
        next!(p)
    end
else
    for i in 1:n
        try
            dicts[i] = extract_data(read_json(files[i]))
        catch e
            @warn "Error in sample at index $i."
        end
    end
end
elapsed = time() - t



function load_samples(d::Dataset; inds = :)
    files = d.samples[inds]
    n = length(files)
    dicts = Vector{ProductNode}(undef, n)

    if typeof(inds) == Colon || length(inds) > 32
        p = Progress(n; desc="Extracting JSONs: ")
        Threads.@threads for i in 1:n
            dicts[i] = d.extractor(read_json(files[i]))
            next!(p)
        end
    else
        for i in 1:n
            try
                dicts[i] = d.extractor(read_json(files[i]))
            catch e
                @warn "Error in sample at index $i."
            end
        end
    end
    return reduce(catobs, dicts)
end

@time samples, labels = load_samples(d, inds=1:64)

#################################################################################################################
###############################################   Loading files   ###############################################
#################################################################################################################

# load benign files
benign_folder = "/mnt/data/jsonlearning/garcia/reports/benign"

folders = 0:9
files = String[]

for f in folders
    onepath = joinpath(benign_folder, string(f))
    subfolders = readdir(onepath)
    filter!(x -> x != "latest", subfolders)
    for s in subfolders
        filepath = joinpath(onepath, s, "report.json")
        if isfile(filepath)
            push!(files, filepath)
        end
    end
end

benign_files = files

# load malicious files
malicious_folder = "/mnt/data/jsonlearning/garcia/reports/malicious"
malicious_files = String[]

for f in folders
    onepath = joinpath(malicious_folder, string(f))
    subfolders = readdir(onepath)
    if subfolders == ["analyses"]
        onepath = joinpath(malicious_folder, string(f), "analyses")
        subfolders = readdir(onepath)
    end
    filter!(x -> x != "latest", subfolders)
    for s in subfolders
        filepath = joinpath(onepath, s, "report.json")
        if isfile(filepath)
            push!(malicious_files, filepath)
        end
    end
end

# save to dataframe which can be used for loading mechanics

labels = vcat(
    repeat(["benign"], length(benign_files)),
    repeat(["malicious"], length(malicious_files))
)

df = DataFrame(
    :samples => vcat(benign_files, malicious_files),
    :classification_family => labels,
    :classification_type => labels,
    :date => repeat([missing], length(labels))
)
CSV.write("/mnt/data/jsonlearning/garcia/reports/labels.csv", df)

########################################################################################################################
############################################### Schema, extractors, etc. ###############################################
########################################################################################################################

# sch = BSON.load(datadir("schema_benign.bson"))[:schema]
sch = BSON.load(datadir("schema.bson"))[:schema]

extractor = suggestextractor(sch)

inds = 1:256
chosen_files = files[inds]
n = length(chosen_files)
dicts = Vector{ProductNode}(undef, n)

p = Progress(n; desc="Extracting JSONs: ")
Threads.@threads for i in 1:n
    dicts[i] = extractor(read_json(chosen_files[i]))
    next!(p)
end

reduce(catobs, dicts)


shahs = Vector{String}(undef, length(files))

@showprogress for (i, x) in enumerate(files)
    try
        js = read_json(x)
        s = js["target"]["file"]["sha256"]
        shahs[i] = s
    catch e
        @warn "Some error occured."
    end
end


for i in 1:100
    js = read_json(files[i])
    println(js["target"]["category"])
end

for i in 1:100
    js = read_json(files[i])
    sev = []
    for k in js["signatures"]
        push!(sev, k["severity"])
    end
    println(sev)
end
js["signatures"][1]