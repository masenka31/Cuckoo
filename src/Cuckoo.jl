module Cuckoo

using Mill
using JSON, JsonGrinder

include("dataset.jl")
include("paths.jl")
include("constructors.jl")

export classifier_constructor
export Dataset

end