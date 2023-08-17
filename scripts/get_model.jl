using Cuckoo
using CSV
using DataFrames
using Cuckoo: _calculate_accuracy, parse_string3
using BSON

dataset = "garcia"
feature_model = "hmil_ts"
end_mdoel = "dense_classifier"
results_dir = expdir("results", dataset, feature_model, end_model) # results directory
files = readdir(results_dir)
ids = unique([x[1:end-5] for x in files if endswith(x, ".bson")])

max_val_accuracy = 0
saved_model = []
R = []
@showprogress for id in ids
    try
        # load parameters
        b = BSON.load(joinpath(results_dir, "$id.bson"))
        # delete model from the file (too large)
        # delete!(b, :model)
        # create a parameters dataframe from it
        bdf = DataFrame(b)

        # load the results and calculate metrics
        c = CSV.read(joinpath(results_dir, "$id.csv"), DataFrame)

        if dataset == "garcia" && feature_model in ["hmil", "hmil_ts"]
            if "2" in c.predicted
                max_value = "2" 
            else
                max_value = "1"
            end
            c.predicted = Int64.(map(x -> parse_string3(x, max_value), c.predicted))
            filter!(:predicted => x -> x < 3, c)
        end

        train_acc, validation_acc, test_acc = _calculate_accuracy(c) # accuracy
        
        if validation_acc > max_val_accuracy
            max_val_accuracy = validation_acc
            push!(saved_model, b[:model])
        end
    catch e
        @warn "File with id $id corrupted, skipping loading..."
    end
end