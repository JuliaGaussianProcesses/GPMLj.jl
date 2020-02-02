module GPJ

using PyCall, Distances

export
    gpflow,
    instantiate!,
    minimize!,
    predict_f,
    predict_f_samples,
    GPFlowObject


abstract type GPFlowObject end

function instantiate!(o) end
function instantiate!(o::Nothing) return nothing end
function minimize!(opt, m) end
function predict_f(m, Xnew) end
function predict_f_samples(m, Xnew, num_samples) end

function elementwise end

const pw = pairwise
const ew = elementwise

# Load necessary utilities
include(joinpath("utils", "abstract_data_set.jl"))


include("kernels.jl")

# Load the GPFlow Interface
include("gpflow.jl")


end # module
