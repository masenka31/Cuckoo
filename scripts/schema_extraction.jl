using DrWatson
@quickactivate

using JsonGrinder
using JSON
using Mill
using ThreadTools
using BSON
using CSV, DataFrames
using ProgressMeter

const datapath = "/mnt/data/jsonlearning/Avast_cockoo"
const datapath_full = "/mnt/data/jsonlearning/Avast_cuckoo_full"
read_json(file) = JSON.parse(read(file, String))

# get the schema for jsons
# sch = schema(readdir("$datapath/public_small_reports", join = true)) do s
sch = schema(readdir("$datapath_full/public_full_reports", join = true)) do s
	open(s,"r") do fio
		read(fio, String)
	end |> JSON.parse
end

wsave("data/schema_full.bson", Dict(:schema => sch))