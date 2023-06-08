# exports

export experiments_path, results_path, splits_path
export cuckoo_path, cuckoo_full_path, garcia_path
export benign_path, malicious_path

export expdir, resultsdir, splitsdir

# generate the paths for data on the cluster

const experiments_path = "/mnt/data/jsonlearning/experiments"
const results_path = "/mnt/data/jsonlearning/results"
const splits_path = "/mnt/data/jsonlearning/splits"

const cuckoo_path = "/mnt/data/jsonlearning/Avast_cuckoo"
const cuckoo_full_path = "/mnt/data/jsonlearning/Avast_cuckoo_full"
const garcia_path = "/mnt/data/jsonlearning/datasets/garcia/reports"

const benign_path = "/mnt/data/jsonlearning/garcia/reports/benign"
const malicious_path = "/mnt/data/jsonlearning/garcia/reports/malicious"

# DrWatson-like functions to work with the paths

expdir(args...) = joinpath(experiments_path, args...)
resultsdir(args...) = joinpath(results_path, args...)
splitsdir(args...) = joinpath(splits_path, args...)