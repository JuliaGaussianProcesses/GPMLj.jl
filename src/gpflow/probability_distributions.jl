module probability_distributions

using ..gpflow
import ..gpflow: instantiate!

export
    Gaussian,
    LogNormal,
    Gamma,
    Laplace,
    Uniform,
    instantiate!,
    logp,
    sample

function logp(obj::T, x) where T <: AbstractProbabilityDistribution
    return  obj.o.logp(x)
end

function sample(obj::T; shape=(1,)) where T <: AbstractProbabilityDistribution
    return  obj.o.sample(;shape=shape)
end

mutable struct Gaussian{T1,T2} <: AbstractProbabilityDistribution
    mu::T1
    var::T2
    o::Union{PyObject, Nothing}
end

function Gaussian(mu, var)
    out = Gaussian(mu, var, nothing)
    instantiate!(out)
    out
end

function instantiate!(o::Union{Gaussian, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.probability_distributions.Gaussian(o.mu, o.var)
    return o.o
end

mutable struct DiagonalGaussian{T1,T2} <: AbstractProbabilityDistribution
    mu::T1
    var::T2
    o::Union{PyObject, Nothing}
end

function DiagonalGaussian(mu, var)
    out = DiagonalGaussian(mu, var, nothing)
    instantiate!(out)
    out
end

function instantiate!(o::Union{DiagonalGaussian, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.probability_distributions.DiagonalGaussian(o.mu, o.var)
    return o.o
end

mutable struct MarkovGaussian{T1,T2} <: AbstractProbabilityDistribution
    mu::T1
    var::T2
    o::Union{PyObject, Nothing}
end

function MarkovGaussian(mu, var)
    out = MarkovGaussian(mu, var, nothing)
    instantiate!(out)
    out
end

function instantiate!(o::Union{MarkovGaussian, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.probability_distributions.MarkovGaussian(o.mu, o.var)
    return o.o
end
end # module
