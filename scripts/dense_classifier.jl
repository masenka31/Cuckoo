using DrWatson
using Flux
using Flux: @epochs

include(srcdir("paths.jl"))
include(srcdir("dataset.jl"))
include(srcdir("data.jl"))
include(srcdir("utils.jl"))

feature_file = ARGS[1]
labels_file = datadir("labels.csv")
seed = 1
tr_ratio = 60
const labelnames = sort(unique(tr_l))

tr_x, tr_l, tr_h, val_x, val_l, val_h, test_x, test_l, test_h = load_split_features(
    feature_file, labels_file,
    seed=seed,
    tr_ratio=tr_ratio
)

idim = size(tr_x, 1)
activation = relu
hdim = 64
odim = 10
nlayers = 3
batchsize = 64

if nlayers == 2
    model = Chain(Dense(idim, hdim, activation), Dense(hdim, odim))
elseif nlayers == 3
    model = Chain(Dense(idim, hdim, activation), Dense(hdim, hdim, activation), Dense(hdim, odim))
else
    model = Chain(Dense(idim, hdim, activation), Dense(hdim, hdim, activation), Dense(hdim, hdim, activation), Dense(hdim, odim))
end

loss(x, y) = Flux.logitcrossentropy(model(x), y)
opt = ADAM()
ps = Flux.params(model)

tr_y = Flux.onehotbatch(tr_l, labelnames)
val_y = Flux.onehotbatch(val_l, labelnames)
train_data = Flux.Data.DataLoader((tr_x, tr_y), batchsize=batchsize)

start_time = time()
max_train_time = 60*15 # 20 minutes of training time
while time() - start_time < max_train_time
    Flux.train!(loss, ps, train_data, opt)
    # acc = loss(val_x, val_y)
    # @info "validation accuracy = $(round(acc, digits=3))"
end

# results

predictions = vcat(
    Flux.onecold(model(tr_x), labelnames)
    Flux.onecold(model(val_x), labelnames)
    Flux.onecold(model(test_x), labelnames)
)

results_df = DataFrame(
    :hash = vcat(train_h, val_h, test_h),
    :ground_truth = vcat(tr_l, val_l, test_l),
    :predicted = predictions,
    :split => vcat(
        repeat(["train"], length(train_h)),
        repeat(["validation"], length(val_h)),
        repeat(["test"], length(test_h))
    )
)

safesave(expdir("results.csv"), results_df)