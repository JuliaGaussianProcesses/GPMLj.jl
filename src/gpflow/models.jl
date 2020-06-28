module models

using ..gpflow
import ..gpflow: instantiate!, predict_f, predict_f_samples
export 
    GPR,
    SGPR,
    VGP,
    SVGP,
    GPMC,
    SGPMC,
    instantiate!,
    predict_f,
    predict_f_samples

abstract type GPModel <: Model end

function predict_f(m::GPModel, Xnew)
    return m.o.predict_f(Xnew)
end
function predict_f_samples(m::GPModel, Xnew, num_samples)
    return m.o.predict_f_samples(Xnew, num_samples)
end

"""
    Gaussian Process Regression.

    This is a vanilla implementation of GP regression with a Gaussian
    likelihood. In this case inference is exact, but costs O(N^3). This means
    that we can compute the predictive distributions (predict_f, predict_y) in
    closed-form, as well as the marginal likelihood, which we use to estimate
    (optimize) the kernel parameters. 
    
    Multiple columns of Y are treated independently, using the same kernel. 

    The log likelihood of this model is sometimes referred to as the
    'marginal log likelihood', and is given by

    .. math::

       \\log p(\\mathbf y | \\mathbf f) = \\mathcal N(\\mathbf y | 0, \\mathbf K + \\sigma_n \\mathbf I)
"""
mutable struct GPR{T} <: GPModel
    data::T
    kernel::Union{Kernel, Nothing}
    mean_function::Union{AbstractMeanFunction, Nothing}
    noise_variance::Real
    o::Union{PyObject, Nothing}
end

function GPR(
    data::Tuple{AbstractArray, AbstractArray},
    kernel::Kernel; 
    mean_function::Union{AbstractMeanFunction, Nothing}=nothing, 
    noise_variance::Real=1.0
)
    out = GPR(data, kernel, mean_function, noise_variance, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{GPR, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    kernel_ = instantiate!(o.kernel)
    mean_function_ = instantiate!(o.mean_function)
    o.o = py_gpflow.models.GPR(o.data, kernel_, mean_function_, o.noise_variance)
    #= TODO: Shoudld we manually initialize likelihood?
    o.likelihood = likelihoods.Gaussian()
    o.likelihood.o = o.o.likelihood =#
    return o.o
end


# TODO: SGPRUpperMixin required?
"""
    Sparse Variational GP regression. The key reference is

    ::

      @inproceedings{titsias2009variational,
        title={Variational learning of inducing variables in
               sparse Gaussian processes},
        author={Titsias, Michalis K},
        booktitle={International Conference on
                   Artificial Intelligence and Statistics},
        pages={567--574},
        year={2009}
      }

"""
mutable struct SGPR{T1,T2,T3} <: GPModel
    X::T1
    Y::T2
    kern::Union{Kernel, Nothing}
    feat
    mean_function::Union{AbstractMeanFunction, Nothing}
    Z::T3
    name::Union{String, Nothing}
    o::Union{PyObject, Nothing}
end

function SGPR(  
    X, 
    Y, 
    kern::Union{Kernel, Nothing};        
    feat=nothing,
    mean_function::Union{AbstractMeanFunction, Nothing}=nothing,
    Z=nothing,
    name::Union{String, Nothing}=nothing
)
    out = SGPR(X, Y, kern, feat, mean_function, Z, name, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{SGPR, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    kern_ = instantiate!(o.kern)
    mean_function_ = instantiate!(o.mean_function)
    o.o = py_gpflow.models.SGPR(
        o.X, o.Y, kern=kern_;
        mean_function=mean_function_,
        name=o.name
    )
    return o.o
end

"""
    This method approximates the Gaussian process posterior using a multivariate Gaussian.

    The idea is that the posterior over the function-value vector F is
    approximated by a Gaussian, and the KL divergence is minimised between
    the approximation and the posterior.

    This implementation is equivalent to svgp with X=Z, but is more efficient.
    The whitened representation is used to aid optimization.

    The posterior approximation is

    .. math::

       q(\\mathbf f) = N(\\mathbf f \\,|\\, \\boldsymbol \\mu, \\boldsymbol \\Sigma)

"""
mutable struct VGP{T} <: GPModel
    data::T
    kernel::Union{Kernel, Nothing}
    likelihood::Union{AbstractLikelihood, Nothing}
    mean_function::Union{AbstractMeanFunction, Nothing}
    num_latent_gps::Union{Int, Nothing}
    o::Union{PyObject, Nothing}
end

function VGP(
    data::Tuple{AbstractArray, AbstractArray},
    kernel::Union{Kernel, Nothing}, 
    likelihood::Union{AbstractLikelihood, Nothing};
    mean_function::Union{AbstractMeanFunction, Nothing}=nothing,
    num_latent_gps::Union{Int, Nothing}=nothing
)
    out = VGP(data, kernel, likelihood, mean_function, num_latent_gps, nothing)
    instantiate!(out)
    return out
end    

function instantiate!(o::VGP)
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    kernel_ = instantiate!(o.kernel)
    likelihood_ = instantiate!(o.likelihood)
    mean_function_ = instantiate!(o.mean_function)
    o.o = py_gpflow.models.VGP(
        o.data, kernel=kernel_, likelihood=likelihood_;
        mean_function=mean_function_,
        num_latent_gps=o.num_latent_gps
    )
    return o.o
end

"""
    This is the Sparse Variational GP (SVGP). The key reference is

    ::

    @inproceedings{hensman2014scalable,
        title={Scalable Variational Gaussian Process Classification},
        author={Hensman, James and Matthews,
                Alexander G. de G. and Ghahramani, Zoubin},
        booktitle={Proceedings of AISTATS},
        year={2015}
    }

"""
mutable struct SVGP{T1,T2,T3,T4,T5} <: GPModel
    data::T1
    kern::Union{Kernel, Nothing}
    likelihood::Union{AbstractLikelihood, Nothing}
    feat::T2
    mean_function::Union{AbstractMeanFunction, Nothing}
    num_latent::Union{Int, Nothing}
    q_diag::Bool
    whiten::Bool
    minibatch_size::Union{Int, Nothing}
    Z::T3
    num_data::Union{Int, Nothing}
    q_mu::T4
    q_sqrt::T5
    o::Union{PyObject, Nothing}
end

function SVGP(
    X, 
    Y, 
    kern::Union{Kernel, Nothing}, 
    likelihood::Union{AbstractLikelihood, Nothing};
    feat=nothing, 
    mean_function::Union{AbstractMeanFunction, Nothing}=nothing, 
    num_latent::Union{Int, Nothing}=nothing, 
    q_diag::Bool=false,
    whiten::Bool=true, 
    minibatch_size::Union{Int, Nothing}=nothing, 
    Z=nothing, 
    num_data=nothing, 
    q_mu=nothing, 
    q_sqrt=nothing
)

    out = SVGP( 
        X, 
        Y, 
        kern, 
        likelihood,
        feat,
        mean_function,
        num_latent,
        q_diag,
        whiten,
        minibatch_size,
        Z,
        num_data,
        q_mu,
        q_sqrt,
        nothing
    )
    instantiate!(out)
    return out
end

function instantiate!(o::SVGP)
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    kern_ = instantiate!(o.kern)
    likelihood_ = instantiate!(o.likelihood)
    mean_function_ = instantiate!(o.mean_function)
    o.o = py_gpflow.models.SGVP(
        o.X, 
        o.Y, 
        kern=kern_, 
        likelihood=likelihood_;
        feat=o.feat,
        mean_function=mean_function_,
        num_latent=o.num_latent,
        q_diag=o.q_diag,
        whiten=o.whiten,
        minibatch_size=o.minibatch_size,
        Z=o.Z,
        num_data=o.num_data,
        q_mu=o.q_mu,
        q_sqrt=o.q_sqrt,
    )
    return o.o
end

mutable struct GPMC{T} <: GPModel
    data::T
    kernel::Union{Kernel, Nothing}
    likelihood::Union{AbstractLikelihood, Nothing}
    mean_function::Union{AbstractMeanFunction, Nothing}
    num_latent_gps::Union{Int, Nothing}
    o::Union{PyObject, Nothing}
end

function GPMC(  
    data::Tuple{AbstractArray, AbstractArray},
    kernel::Union{Kernel, Nothing}, 
    likelihood::Union{AbstractLikelihood, Nothing};
    mean_function::Union{AbstractMeanFunction, Nothing}=nothing, 
    num_latent_gps::Union{Int, Nothing}=nothing,
)
    out = GPMC( 
        data,
        kernel,
        likelihood,
        mean_function,
        num_latent_gps,
        nothing
    )
    instantiate!(out)
    return out
end

function instantiate!(o::GPMC)
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    kernel_ = instantiate!(o.kernel)
    likelihood_ = instantiate!(o.likelihood)
    mean_function_ = instantiate!(o.mean_function)
    o.o = py_gpflow.models.GPMC(
        o.data, 
        kernel=kernel_, 
        likelihood=likelihood_;
        mean_function=mean_function_,
        num_latent_gps=o.num_latent_gps                                
    )
    return o.o
end


"""
    This is the Sparse Variational GP using MCMC (SGPMC). The key reference is

    ::

      @inproceedings{hensman2015mcmc,
        title={MCMC for Variatinoally Sparse Gaussian Processes},
        author={Hensman, James and Matthews, Alexander G. de G.
                and Filippone, Maurizio and Ghahramani, Zoubin},
        booktitle={Proceedings of NIPS},
        year={2015}
      }

    The latent function values are represented by centered
    (whitened) variables, so

    .. math::
       :nowrap:

       \\begin{align}
       \\mathbf v & \\sim N(0, \\mathbf I) \\\\
       \\mathbf u &= \\mathbf L\\mathbf v
       \\end{align}

    with

    .. math::
        \\mathbf L \\mathbf L^\\top = \\mathbf K


"""
mutable struct SGPMC{T1,T2,T3,T4} <: GPModel
    X::T1
    Y::T2
    kern::Union{Kernel, Nothing}
    likelihood::Union{AbstractLikelihood, Nothing}
    feat::T3
    mean_function::Union{AbstractMeanFunction, Nothing}
    num_latent::Union{Int, Nothing}
    Z::T4
    o::Union{PyObject, Nothing}
end

function SGPMC( 
    X, 
    Y, 
    kern::Union{Kernel, Nothing}, 
    likelihood::Union{AbstractLikelihood, Nothing};
    feat=nothing,
    mean_function::Union{AbstractMeanFunction, Nothing}=nothing, 
    num_latent::Union{Int, Nothing}=nothing,
    Z=nothing
    )
    out = SGPMC(
        X,
        Y,
        kern,
        likelihood,
        feat,
        mean_function,
        num_latent,
        Z,
        nothing
    )
    instantiate!(out)
    return out
end

function instantiate!(o::SGPMC)
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    kern_ = instantiate!(o.kern)
    likelihood_ = instantiate!(o.likelihood)
    mean_function_ = instantiate!(o.mean_function)
    o.o = py_gpflow.models.SGPMC(
        o.X, 
        o.Y, 
        kern=kern_, 
        likelihood=likelihood_;
        fwat=o.feat,
        mean_function=mean_function_,
        num_latent=o.num_latent,
        Z=o.Z
    )
    return o.o
end

end # module
