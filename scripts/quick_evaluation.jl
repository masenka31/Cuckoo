using Cuckoo
using DataFrames
using PrettyTables

# HMIL classifier
df = get_combined_results("cuckoo_small", "hmil", "dense_classifier", min_seed=5)
df.train_acc = round.(df.train_acc, digits=3)
df.validation_acc = round.(df.validation_acc, digits=3)
df.test_acc = round.(df.test_acc, digits=3)
first(df, 5)

pretty_table(df[1:5, Not([:tr_ratio, :repetition])], tf=tf_latex_booktabs, nosubheader=true)

# Pedro + classifier
df = get_combined_results("cuckoo_small", "pedro007", "dense_classifier", min_seed=5)
df.train_acc = round.(df.train_acc, digits=3)
df.validation_acc = round.(df.validation_acc, digits=3)
df.test_acc = round.(df.test_acc, digits=3)
first(df, 5)

pretty_table(df[1:5, Not([:classification_model, :ratio, :repetition])], tf=tf_latex_booktabs, nosubheader=true)

# Pedro + xgboost
df = get_combined_results("cuckoo_small", "pedro007", "xgboost", min_seed=5)
df.train_acc = round.(df.train_acc, digits=3)
df.validation_acc = round.(df.validation_acc, digits=3)
df.test_acc = round.(df.test_acc, digits=3)
first(df, 5)

pretty_table(df[1:5, Not([:classification_model, :ratio])], tf=tf_latex_booktabs, nosubheader=true)

# Vasek + classifier
df = get_combined_results("", "vasek007", "dense_classifier", min_seed=5)
df.train_acc = round.(df.train_acc, digits=3)
df.validation_acc = round.(df.validation_acc, digits=3)
df.test_acc = round.(df.test_acc, digits=3)
first(df, 5)

pretty_table(df[1:5, Not([:classification_model, :ratio, :repetition])], tf=tf_latex_booktabs, nosubheader=true)

# Vasek + xgboost
df = get_combined_results("cuckoo_small", "vasek007", "xgboost", min_seed=5)
df.train_acc = round.(df.train_acc, digits=3)
df.validation_acc = round.(df.validation_acc, digits=3)
df.test_acc = round.(df.test_acc, digits=3)
first(df, 5)

pretty_table(df[1:5, Not([:classification_model, :ratio])], tf=tf_latex_booktabs, nosubheader=true)