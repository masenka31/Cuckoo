const experiments_path = "/mnt/data/jsonlearning/experiments"
const results_path = "/mnt/data/jsonlearning/results"
const cuckoo_small_path = "/mnt/data/jsonlearning/Avast_cockoo"
const cuckoo_full_path = "/mnt/data/jsonlearning/Avast_cuckoo_full"
const split_path = "/mnt/data/jsonlearning/splits/Avast_cockoo"

experimentsdir(args...) = joinpath(experiments_path, args...)
expdir(args...) = joinpath(experiments_path, args...)
resultsdir(args...) = joinpath(results_path, args...)
splitdir(args...) = joinpath(split_path, args...)