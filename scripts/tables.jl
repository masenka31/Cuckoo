using Cuckoo
using DataFrames
using PrettyTables
using Dates

cuckoo_results = DataFrame[]

function create_result_dataframe(df, feature_model, classification_model; digits=3)
    d = DataFrame(
        "feature model" => feature_model,
        "classifier" => classification_model,
        "train acc" => round(df[1, :train_acc], digits=digits),
        "val acc" => round(df[1, :validation_acc], digits=digits),
        "test acc" => round(df[1, :test_acc], digits=digits),
    )
    return d
end

# HMIL classifier
df = get_combined_results("cuckoo_small", "hmil", "dense_classifier", min_seed=5)
push!(cuckoo_results, create_result_dataframe(df, "HMIL", "NN classifier"))

# Pedro + classifier
df = get_combined_results("cuckoo_small", "pedro007", "dense_classifier", min_seed=5)
push!(cuckoo_results, create_result_dataframe(df, "Pedro", "NN classifier"))

# Pedro + classifier with dropout
df = get_combined_results("", "pedro", "dense_classifier", min_seed=5)
push!(cuckoo_results, create_result_dataframe(df, "Pedro", "NN classifier (+ dropout)"))

# Pedro + XGBoost
df = get_combined_results("cuckoo_small", "pedro007", "xgboost", min_seed=5)
push!(cuckoo_results, create_result_dataframe(df, "Pedro", "XGBoost"))

# Vasek + classifier
df = get_combined_results("", "vasek007", "dense_classifier", min_seed=5)
push!(cuckoo_results, create_result_dataframe(df, "Vasek", "NN classifier"))

# Vasek + XGBoost
df = get_combined_results("cuckoo_small", "vasek007", "xgboost", min_seed=5)
push!(cuckoo_results, create_result_dataframe(df, "Vasek", "XGBoost"))

CR = vcat(cuckoo_results...)
sort!(CR, "test acc", rev=true)

pretty_table(CR, tf=tf_latex_booktabs, nosubheader=true)

########################################################################################
###                                      Garcia                                      ###
########################################################################################

garcia_results = DataFrame[]

# HMIL
df = get_combined_results("garcia", "hmil", "dense_classifier", min_seed=5)
push!(garcia_results, create_result_dataframe(df, "HMIL", "NN classifier"))

# Pedro + classifier
df = get_combined_results("garcia", "pedro", "dense_classifier", min_seed=5)
push!(garcia_results, create_result_dataframe(df, "Pedro", "NN classifier"))

# Pedro + classifier + dropout
df = get_combined_results("garcia", "pedro_dropout", "dense_classifier", min_seed=5)
push!(garcia_results, create_result_dataframe(df, "Pedro", "NN classifier (+ dropout)"))

# Pedro + xgboost
df = get_combined_results("garcia", "pedro", "xgboost", min_seed=10)
push!(garcia_results, create_result_dataframe(df, "Pedro", "XGBoost"))

GR = vcat(garcia_results...)
sort!(GR, "test acc", rev=true)

pretty_table(GR, tf=tf_latex_booktabs, nosubheader=true)