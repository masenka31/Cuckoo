using Plots
ENV["GKSwstype"] = "100"

function scatter2(X, x=1, y=2; kwargs...)
    scatter(X[x,:],X[y,:]; label="", kwargs...)
    savefig("plot.png")
end
function scatter2!(X, x=1, y=2; kwargs...)
    scatter!(X[x,:],X[y,:]; label="", kwargs...)
    savefig("plot.png")
end

function scatter3(X, x=1, y=2, z=3; kwargs...)
    scatter(X[x,:],X[y,:],X[z,:]; label="", kwargs...)
    savefig("plot.png")
end
function scatter3!(X, x=1, y=2, z=3; kwargs...)
    scatter!(X[x,:],X[y,:],X[z,:]; label="", kwargs...)
    savefig("plot.png")
end