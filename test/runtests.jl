using GPJ, Test, Random

Random.seed!(123456)

include("kernels.jl")
include("gpflow/gpflow.jl")
