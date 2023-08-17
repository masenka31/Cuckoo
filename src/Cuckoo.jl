module Cuckoo

using Mill
using JSON, JsonGrinder

include("dataset.jl")
include("data.jl")
include("paths.jl")
include("constructors.jl")
include("utils.jl")
include("evaluation.jl")

export hmil_classifier_constructor, dense_classifier_constructor
export Dataset
export load_split_features, load_split_indexes, load_indexes
export encode_labels
export get_results, get_combined_results
export cuckoo_hash_to_int, cuckoo_int_to_hash
export garcia_hash_to_int, garcia_int_to_hash

end