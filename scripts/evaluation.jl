using DrWatson
using DataFrames, CSV
using Statistics
using PrettyTables
include(srcdir("paths.jl"))
include(srcdir("evaluation.jl"))

pedro_df = load_results(expdir("cuckoo_small", "pedro007", "dense_classifier"), "pedro007")
misa_df = load_results(expdir("cuckoo_small", "hmill_classifier", "dense_classifier"), "hmill_classifier")

### creating tables
# pedro_df = vcat(tr_g...)
pedro_df = vcat(map(i -> DataFrame(pedro_df[i][1,:]), 1:4)...)
misa_df = vcat(map(i -> DataFrame(misa_df[i][1,:]), 1:4)...)

fn = [:tr_ratio, :train, :validation, :test]
f = filter(:tr_ratio => x -> any(x .== [60, "time57", "time41", 20]), vcat(pedro_df[:, fn],misa_df[:, fn]))

f[!, :model] = vcat(repeat(["pedro\'s features"], 4), repeat(["HMill + classifier"], 4))

f[!, :dataset_type] = ["60/20/20 split", "20/40/40 split", "time split (57% train)", "time split (41% train)","60/20/20 split",  "20/40/40 split", "time split (57% train)", "time split (41% train)"]

sort!(f, :dataset_type)

pretty_table(f[:, [:dataset_type, :model, :train, :validation, :test]], nosubheader=true, formatters = ft_printf("%5.3f"), hlines=[0,1,3,5,7,9])
pretty_table(f[:, [:dataset_type, :model, :train, :validation, :test]], nosubheader=true, formatters = ft_printf("%5.3f"), hlines=[0,1,3,5,7,9], tf=tf_latex_booktabs)