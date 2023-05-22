using DrWatson
using Cuckoo
using JSON, BSON, JsonGrinder, Mill

using Flux
using UUIDs

# schema preparation and extraction

d = Dataset("garcia")

using JsonGrinder, Flux, Mill, MLDataPattern, JSON, HierarchicalUtils, StatsBase, OrderedCollections
using JsonGrinder: DictEntry, Entry

sch = schema(d.samples) do s
	open(s,"r") do fio
		read(fio, String)
	end |> JSON.parse
end

safesave(datadir("garcia_schema_run.bson"), Dict(:schema => sch))
generate_html("garcia_schema_run.html", sch, max_vals=nothing)