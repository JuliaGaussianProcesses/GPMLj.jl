module likelihoods

    using ..gpflow
    import ..gpflow: compile!
    export 
        Gaussian,
        compile!

    mutable struct Gaussian{T}<:Likelihood
        variance::T
        o::Union{PyObject,Nothing}
    end

    function Gaussian(;variance::T=1.0) where T <: Real
        Gaussian(variance, nothing)
    end

    function compile!(o::Union{Gaussian,Nothing})
        if typeof(o.o)<:PyObject return o.o end
        if o === nothing return nothing end
        o.o = py_gpflow.likelihoods.Gaussian(;variance=o.variance)
    end

end # module
