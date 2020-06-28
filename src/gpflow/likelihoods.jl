module likelihoods

using ..gpflow
import ..gpflow: instantiate!
export 
    Bernoulli,
    Beta,
    Exponential,
    Gamma,
    Gaussian,
    Likelihood,
    MonteCarloLikelihood,
    MultiClass,
    Ordinal,
    Poisson,
    RobustMax,
    Softmax,
    StudentT,
    SwitchedLikelihood,
    instantiate!

mutable struct Bernoulli<:AbstractLikelihood
    # invlink # TODO: add capability to add custom invlink
    o::Union{PyObject, Nothing}
end

function Bernoulli()
    out = Bernoulli(nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Bernoulli, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.likelihoods.Bernoulli()
end

"""
    This uses a reparameterisation of the Beta density. We have the mean of the
    Beta distribution given by the transformed process:

        m = sigma(f)

    and a scale parameter. The familiar alpha, beta parameters are given by

        m     = alpha / (alpha + beta)
        scale = alpha + beta

    so:
        alpha = scale * m
        beta  = scale * (1-m)
"""
mutable struct Beta{T}<:AbstractLikelihood
    # invlink # TODO: add capability to add custom invlink
    scale::T
    o::Union{PyObject, Nothing}
end

function Beta(;scale::Real=1.0)
    out = Beta(scale, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Beta, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.likelihoods.Beta(;scale=o.scale)
end

mutable struct Exponential<:AbstractLikelihood
    # invlink # TODO: add capability to add custom invlink
    o::Union{PyObject, Nothing}
end

function Exponential()
    out = Exponential(nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Exponential, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.likelihoods.Exponential()
end

"""
    Use the transformed GP to give the *scale* (inverse rate) of the Gamma
"""
mutable struct Gamma<:AbstractLikelihood
    # invlink # TODO: add capability to add custom invlink
    o::Union{PyObject, Nothing}
end

function Gamma()
    out = Gamma(nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Gamma, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.likelihoods.Gamma()
end

mutable struct Gaussian{T}<:AbstractLikelihood
    variance::T
    o::Union{PyObject, Nothing}
end

function Gaussian(;variance::T=1.0) where T <: Real
    out = Gaussian(variance, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Gaussian, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.likelihoods.Gaussian(;variance=o.variance)
end

# TODO GaussianMC not implemented in GPFlow
"""
    A likelihood for doing ordinal regression.

    The data are integer values from 0 to K, and the user must specify (K-1)
    'bin edges' which define the points at which the labels switch. Let the bin
    edges be [a_0, a_1, ... a_{K-1}], then the likelihood is

    p(Y=0|F) = phi((a_0 - F) / sigma)
    p(Y=1|F) = phi((a_1 - F) / sigma) - phi((a_0 - F) / sigma)
    p(Y=2|F) = phi((a_2 - F) / sigma) - phi((a_1 - F) / sigma)
    ...
    p(Y=K|F) = 1 - phi((a_{K-1} - F) / sigma)

    where phi is the cumulative density function of a Gaussian (the inverse probit
    function) and sigma is a parameter to be learned. A reference is:

    @article{chu2005gaussian,
      title={Gaussian processes for ordinal regression},
      author={Chu, Wei and Ghahramani, Zoubin},
      journal={Journal of Machine Learning Research},
      volume={6},
      number={Jul},
      pages={1019--1041},
      year={2005}
    }
"""
function GaussianMC()
    throw("GaussianMC not implemented")
end

mutable struct Likelihood<:AbstractLikelihood
    latent_dim
    observation_dim
    o::Union{PyObject, Nothing}
end

mutable struct MultiClass<:AbstractLikelihood
    num_classes::Integer
    # invlink # TODO: add capability to add custom invlink
    o::Union{PyObject, Nothing}
end

function MultiClass(num_classes::Integer)
    out = MultiClass(num_classes, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{MultiClass, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.likelihoods.MultiClass(o.num_classes)
end

"""
    A likelihood for doing ordinal regression.

    The data are integer values from 0 to K, and the user must specify (K-1)
    'bin edges' which define the points at which the labels switch. Let the bin
    edges be [a_0, a_1, ... a_{K-1}], then the likelihood is

    p(Y=0|F) = phi((a_0 - F) / sigma)
    p(Y=1|F) = phi((a_1 - F) / sigma) - phi((a_0 - F) / sigma)
    p(Y=2|F) = phi((a_2 - F) / sigma) - phi((a_1 - F) / sigma)
    ...
    p(Y=K|F) = 1 - phi((a_{K-1} - F) / sigma)

    where phi is the cumulative density function of a Gaussian (the inverse probit
    function) and sigma is a parameter to be learned. A reference is:

    @article{chu2005gaussian,
      title={Gaussian processes for ordinal regression},
      author={Chu, Wei and Ghahramani, Zoubin},
      journal={Journal of Machine Learning Research},
      volume={6},
      number={Jul},
      pages={1019--1041},
      year={2005}
    }
"""
mutable struct Ordinal{T}<:AbstractLikelihood
    bin_edges::T
    o::Union{PyObject, Nothing}
end

function Ordinal(bin_edges)
    out = Ordinal(bin_edges, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Ordinal, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.likelihoods.Ordinal(o.bin_edges)
end

"""
    Poisson likelihood for use with count data, where the rate is given by the (transformed) GP.

    let g(.) be the inverse-link function, then this likelihood represents

    p(y_i | f_i) = Poisson(y_i | g(f_i) * binsize)

    Note:binsize
    For use in a Log Gaussian Cox process (doubly stochastic model) where the
    rate function of an inhomogeneous Poisson process is given by a GP.  The
    intractable likelihood can be approximated by gridding the space (into bins
    of size 'binsize') and using this Poisson likelihood.
"""
mutable struct Poisson{T}<:AbstractLikelihood
    # invlink # TODO: add capability to add custom invlink
    binsize::T
    o::Union{PyObject, Nothing}
end

function Poisson(;binsize::Real=1.0)
    out = Poisson(binsize, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Poisson, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.likelihoods.Poisson(;binsize=o.binsize)
end

"""
    This class represent a multi-class inverse-link function. Given a vector
    f=[f_1, f_2, ... f_k], the result of the mapping is

    y = [y_1 ... y_k]

    with

    y_i = (1-eps)  i == argmax(f)
          eps/(k-1)  otherwise.
"""
mutable struct RobustMax{T}<:AbstractLikelihood where T <: Real
    num_classes::Integer
    epsilon::T
    o::Union{PyObject, Nothing}
end

function RobustMax(num_classes::Integer; epsilon::Real=1e-3)
    out = RobustMax(num_classes, epsilon, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{RobustMax, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.likelihoods.RobustMax(o.num_classes; epsilon=o.epsilon)
end

"""
    The soft-max multi-class likelihood.
"""
mutable struct Softmax<:AbstractLikelihood
    num_classes::Integer
    o::Union{PyObject, Nothing}
end

function Softmax(num_classes::Integer)
    out = Softmax(num_classes, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Softmax, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.likelihoods.Softmax(o.num_classes)
end

mutable struct StudentT{T1,T2}<:AbstractLikelihood where {T1 <: Real, T2 <: Real}
    scale::T1
    df::T2
    o::Union{PyObject, Nothing}
end

function StudentT(;scale::Real=1.0, df::Real=3.0)
    out = StudentT(scale, df, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{StudentT, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.likelihoods.StudentT(;scale=o.scale, df=o.df)
end

mutable struct SwitchedLikelihood{T}<:AbstractLikelihood
    likelihood_list::T
    o::Union{PyObject, Nothing}
end

function SwitchedLikelihood(likelihood_list)
    out = SwitchedLikelihood(likelihood_list, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{SwitchedLikelihood, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.likelihoods.SwitchedLikelihood(likelihood_list)
end

end # module
