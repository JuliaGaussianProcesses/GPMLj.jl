module kernels

using ..gpflow
import ..gpflow: instantiate!
export
    Combination,
    Product,
    Sum,
    Stationary,
    Convolutional,
    WeightedConvolutional,
    Cosine,
    Exponential,
    RationalQuadratic,
    SquaredExponential,
    Linear,
    Polynomial,
    Matern12,
    Matern32,
    Matern52,
    Static,
    White,
    Bias,
    Constant,
    ArcCosine,
    Periodic,
    Coregion,
    instantiate!

# Stationary Kernels

abstract type AbstractStationary <: Kernel end
abstract type AbstractLinear <: Kernel end
abstract type AbstractCombination <: Kernel end
abstract type AbstractStatic <: Kernel end


"""
    Combine a list of kernels, e.g. by adding or multiplying (see inheriting
    classes).

    The names of the kernels to be combined are generated from their class
    names.
"""
mutable struct Combination <: AbstractCombination
    kernels
    name::Union{String, Nothing}
    o::Union{PyObject, Nothing}
end

function Combination(
    kernels;
    name=nothing
)
    out = Combination(kernels, name, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Combination, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.kernels.Combination(
        [kern.o for kern in o.kernels];
        name=o.name
    )
    return o.o
end

mutable struct Product <: AbstractCombination
    kernels
    name::Union{String, Nothing}
    o::Union{PyObject, Nothing}
end

function Product(
    kernels;
    name=nothing
)
    out = Product(kernels, name, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Product, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.kernels.Product(
        [kern.o for kern in o.kernels];
        name=o.name
    )
    return o.o
end

mutable struct Sum <: AbstractCombination
    kernels
    name::Union{String, Nothing}
    o::Union{PyObject, Nothing}
end

function Sum(
    kernels;
    name=nothing
)
    out = Sum(kernels, name, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Sum, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.kernels.Sum(
        [kern.o for kern in o.kernels];
        name=o.name
    )
    return o.o
end

"""
    Base class for kernels that are stationary, that is, they only depend on

        r = || x - x' ||

    This class handles 'ARD' behaviour, which stands for 'Automatic Relevance
    Determination'. This means that the kernel has one lengthscale per
    dimension, otherwise the kernel is isotropic (has a single lengthscale).
"""
mutable struct Stationary <: AbstractStationary
    input_dim
    variance
    lengthscales
    active_dims
    ARD
    name::Union{String, Nothing}
    o::Union{PyObject, Nothing}
end

function Stationary(
    input_dim;
    variance=1.0,
    lengthscales=1.0,
    active_dims=nothing,
    ARD=nothing,
    name=nothing
)
    out = Stationary(input_dim, variance, lengthscales, active_dims, ARD, name, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Stationary, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.kernels.Stationary(
        o.input_dim;
        variance=o.variance,
        lengthscales=o.lengthscales,
        active_dims=o.active_dims,
        ARD=o.ARD,
        name=o.name
    )
    return o.o
end

"""
    Plain convolutional kernel as described in \\citet{vdw2017convgp}. Defines
    a GP f( ) that is constructed from a sum of responses of individual patches
    in an image:
    f(x) = \\sum_p x^{[p]}
    where x^{[p]} is the pth patch in the image.

    @incollection{vdw2017convgp,
    title = {Convolutional Gaussian Processes},
    author = {van der Wilk, Mark and Rasmussen, Carl Edward and Hensman, James},
    booktitle = {Advances in Neural Information Processing Systems 30},
    year = {2017},
    url = {http://papers.nips.cc/paper/6877-convolutional-gaussian-processes.pdf}
    }
"""
mutable struct Convolutional <: AbstractStationary
    basekern
    img_size
    patch_size
    colour_channels
    o::Union{PyObject, Nothing}
end

function Convolutional(
    basekern,
    img_size,
    patch_size;
    colour_channels=1
)
    out = Convolutional(basekern, img_size, patch_size, colour_channels, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Convolutional, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.kernels.Convolutional(
        o.basekern,
        o.img_size,
        o.patch_size;
        colour_channels=o.colour_channels
    )
    return o.o
end

"""
    Similar to `Convolutional`, but with different weights for each patch:
      f(x) = \\sum_p x^{[p]}

     @incollection{vdw2017convgp,
      title = {Convolutional Gaussian Processes},
      author = {van der Wilk, Mark and Rasmussen, Carl Edward and Hensman, James},
      booktitle = {Advances in Neural Information Processing Systems 30},
      year = {2017},
      url = {http://papers.nips.cc/paper/6877-convolutional-gaussian-processes.pdf}
    }
"""
mutable struct WeightedConvolutional <: AbstractStationary
    basekern
    img_size
    patch_size
    colour_channels
    o::Union{PyObject, Nothing}
end

function WeightedConvolutional(
    basekern,
    img_size,
    patch_size;
    colour_channels=1
)
    out = WeightedConvolutional(basekern, img_size, patch_size, colour_channels, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{WeightedConvolutional, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.kernels.WeightedConvolutional(
        o.basekern,
        o.img_size,
        o.patch_size;
        colour_channels=o.colour_channels
    )
    return o.o
end

"""
    The Cosine kernel. Functions drawn from a GP with this kernel are sinusoids
    (with a random phase).  The kernel equation is

        k(r) =  σ² cos{r}

    where:
    r  is the Euclidean distance between the input points, scaled by the lengthscale parameter ℓ,
    σ² is the variance parameter.
"""
mutable struct Cosine <: AbstractStationary
    input_dim
    variance
    lengthscales
    active_dims
    ARD
    name::Union{String, Nothing}
    o::Union{PyObject, Nothing}
end

function Cosine(
    input_dim;
    variance=1.0,
    lengthscales=1.0,
    active_dims=nothing,
    ARD=nothing,
    name=nothing
)
    out = Cosine(input_dim, variance, lengthscales, active_dims, ARD, name, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Cosine, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.kernels.Cosine(
        o.input_dim;
        variance=o.variance,
        lengthscales=o.lengthscales,
        active_dims=o.active_dims,
        ARD=o.ARD,
        name=o.name
    )
    return o.o
end

"""
    The Exponential kernel. It is equivalent to a Matern12 kernel with doubled lengthscales.
"""
mutable struct Exponential <: AbstractStationary
    input_dim
    variance
    lengthscales
    active_dims
    ARD
    name::Union{String, Nothing}
    o::Union{PyObject, Nothing}
end

function Exponential(
    input_dim;
    variance=1.0,
    lengthscales=1.0,
    active_dims=nothing,
    ARD=nothing,
    name=nothing
)
    out = Exponential(input_dim, variance, lengthscales, active_dims, ARD, name, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Exponential, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.kernels.Exponential(
        o.input_dim;
        variance=o.variance,
        lengthscales=o.lengthscales,
        active_dims=o.active_dims,
        ARD=o.ARD,
        name=o.name
    )
    return o.o
end

"""
    Rational Quadratic kernel. The kernel equation is

    k(r) = σ² (1 + r² / 2α)^(-α)

    where:
    r  is the Euclidean distance between the input points, scaled by the lengthscale parameter ℓ,
    σ² is the variance parameter,
    α  determines relative weighting of small-scale and large-scale fluctuations.

    For α → ∞, the RQ kernel becomes equivalent to the squared exponential.
"""
mutable struct RationalQuadratic <: AbstractStationary
    input_dim
    variance
    lengthscales
    active_dims
    ARD
    name::Union{String, Nothing}
    o::Union{PyObject, Nothing}
end

function RationalQuadratic(
    input_dim;
    variance=1.0,
    lengthscales=1.0,
    active_dims=nothing,
    ARD=nothing,
    name=nothing
)
    out = RationalQuadratic(input_dim, variance, lengthscales, active_dims, ARD, name, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{RationalQuadratic, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.kernels.RationalQuadratic(
        o.input_dim;
        variance=o.variance,
        lengthscales=o.lengthscales,
        active_dims=o.active_dims,
        ARD=o.ARD,
        name=o.name
    )
    return o.o
end

"""
    The radial basis function (RBF) or squared exponential kernel. The kernel equation is

        k(r) = σ² exp{-½ r²}

    where:
    r   is the Euclidean distance between the input points, scaled by the lengthscale parameter ℓ.
    σ²  is the variance parameter

    Functions drawn from a GP with this kernel are infinitely differentiable!
"""
mutable struct SquaredExponential <: AbstractStationary
    input_dim
    variance
    lengthscales
    active_dims
    ARD
    name::Union{String, Nothing}
    o::Union{PyObject, Nothing}
end

function SquaredExponential(
    input_dim;
    variance=1.0,
    lengthscales=1.0,
    active_dims=nothing,
    ARD=nothing,
    name=nothing
)
    out = SquaredExponential(input_dim, variance, lengthscales, active_dims, ARD, name, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{SquaredExponential, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.kernels.SquaredExponential(
        o.input_dim;
        variance=o.variance,
        lengthscales=o.lengthscales,
        active_dims=o.active_dims,
        ARD=o.ARD,
        name=o.name
    )
    return o.o
end

RBF = SquaredExponential

"""
    The linear kernel. Functions drawn from a GP with this kernel are linear, i.e. f(x) = cx. The kernel equation is

        k(x, y) = σ²xy

    where:
    σ²  is the variance parameter.
"""
mutable struct Linear <: AbstractLinear
    input_dim
    variance
    active_dims
    ARD
    name::Union{String, Nothing}
    o::Union{PyObject, Nothing}
end

function Linear(
    input_dim;
    variance=1.0,
    active_dims=nothing,
    ARD=nothing,
    name=nothing
)
    out = Linear(input_dim, variance, active_dims, ARD, name, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Linear, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.kernels.Linear(
        o.input_dim;
        variance=o.variance,
        active_dims=o.active_dims,
        ARD=o.ARD,
        name=o.name
    )
    return o.o
end

"""
    The Polynomial kernel. Functions drawn from a GP with this kernel are
    polynomials of degree `d`. The kernel equation is

        k(x, y) = (σ²xy + γ) ^ d

    where:
    σ² is the variance parameter,
    γ is the offset parameter,
    d is the degree parameter.
"""
mutable struct Polynomial <: AbstractLinear
    input_dim
    degree
    variance
    offset
    active_dims
    ARD
    name::Union{String, Nothing}
    o::Union{PyObject, Nothing}
end

function Polynomial(
    input_dim;
    degree=3.0,
    variance=1.0,
    offset=1.0,
    active_dims=nothing,
    ARD=nothing,
    name=nothing
)
    out = Polynomial(input_dim, degree, variance, offset, active_dims, ARD, name, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Polynomial, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.kernels.Polynomial(
        o.input_dim;
        degree=o.degree,
        variance=o.variance,
        offset=o.offset,
        active_dims=o.active_dims,
        ARD=o.ARD,
        name=o.name
    )
    return o.o
end

"""
    The Matern 1/2 kernel. Functions drawn from a GP with this kernel are not
    differentiable anywhere. The kernel equation is

    k(r) = σ² exp{-r}

    where:
    r  is the Euclidean distance between the input points, scaled by the lengthscale parameter ℓ.
    σ² is the variance parameter
"""
mutable struct Matern12 <: AbstractStationary
    variance
    lengthscales
    active_dims
    name::Union{String, Nothing}
    o::Union{PyObject, Nothing}
end

function Matern12(
    variance=1.0,
    lengthscales=1.0,
    active_dims=nothing,
    name=nothing
)
    out = Matern12(variance, lengthscales, active_dims, name, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Matern12, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.kernels.Matern12(
        variance=o.variance,
        lengthscales=o.lengthscales,
        active_dims=o.active_dims,
        name=o.name
    )
    return o.o
end

mutable struct Matern32 <: AbstractStationary
    variance
    lengthscales
    active_dims
    name::Union{String, Nothing}
    o::Union{PyObject, Nothing}
end

"""
    The Matern 3/2 kernel. Functions drawn from a GP with this kernel are once
    differentiable. The kernel equation is

    k(r) = σ² (1 + √3r) exp{-√3 r}

    where:
    r  is the Euclidean distance between the input points, scaled by the lengthscale parameter ℓ,
    σ² is the variance parameter.
"""
function Matern32(
    variance=1.0,
    lengthscales=1.0,
    active_dims=nothing,
    name=nothing
)
    out = Matern32(variance, lengthscales, active_dims, name, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Matern32, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.kernels.Matern32(
        variance=o.variance,
        lengthscales=o.lengthscales,
        active_dims=o.active_dims,
        name=o.name
    )
    return o.o
end

"""
    The Matern 5/2 kernel. Functions drawn from a GP with this kernel are twice
    differentiable. The kernel equation is

    k(r) = σ² (1 + √5r + 5/3r²) exp{-√5 r}

    where:
    r  is the Euclidean distance between the input points, scaled by the lengthscale parameter ℓ,
    σ² is the variance parameter.
"""
mutable struct Matern52 <: AbstractStationary
    variance
    lengthscales
    active_dims
    name::Union{String, Nothing}
    o::Union{PyObject, Nothing}
end

function Matern52(
    variance=1.0,
    lengthscales=1.0,
    active_dims=nothing,
    name=nothing
)
    out = Matern52(variance, lengthscales, active_dims, name, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Matern52, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.kernels.Matern52(
        variance=o.variance,
        lengthscales=o.lengthscales,
        active_dims=o.active_dims,
        name=o.name
    )
    return o.o
end

# Static Kernels
"""
    Kernels who don't depend on the value of the inputs are 'Static'.  The only
    parameter is a variance, σ².
"""
mutable struct Static <: AbstractStatic
    variance
    active_dims
    o::Union{PyObject, Nothing}
end

function Static(
    variance=1.0,
    active_dims=nothing,
)
    out = Static(variance, active_dims, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Static, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.kernels.Static(
        variance=o.variance,
        active_dims=o.active_dims,
    )
    return o.o
end

"""
    The White kernel: this kernel produces 'white noise'. The kernel equation is

        k(x_n, x_m) = δ(n, m) σ²

    where:
    δ(.,.) is the Kronecker delta,
    σ²  is the variance parameter.
"""
mutable struct White <: AbstractStatic
    variance
    active_dims
    o::Union{PyObject, Nothing}
end

function White(;
    variance=1.0,
    active_dims=nothing,
)
    out = White(variance, active_dims, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{White, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.kernels.White(;
        variance=o.variance,
        active_dims=o.active_dims,
    )
    return o.o
end

"""
    Another name for the Constant kernel, included for convenience.
"""
mutable struct Bias <: AbstractStatic
    variance
    active_dims
    o::Union{PyObject, Nothing}
end

function Bias(;
    variance=1.0,
    active_dims=nothing,
)
    out = Bias(variance, active_dims, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Bias, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.kernels.Bias(;
        variance=o.variance,
        active_dims=o.active_dims,
    )
    return o.o
end

"""
    The Constant (aka Bias) kernel. Functions drawn from a GP with this kernel
    are constant, i.e. f(x) = c, with c ~ N(0, σ^2). The kernel equation is

        k(x, y) = σ²

    where:
    σ²  is the variance parameter.
"""
mutable struct Constant <: AbstractStatic
    variance
    active_dims
    o::Union{PyObject, Nothing}
end

function Constant(;
    variance=1.0,
    active_dims=nothing,
)
    out = Constant(variance, active_dims, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Constant, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.kernels.Constant(;
        variance=o.variance,
        active_dims=o.active_dims,
    )
    return o.o
end

"""
    The Arc-cosine family of kernels which mimics the computation in neural
    networks. The order parameter specifies the assumed activation function.
    The Multi Layer Perceptron (MLP) kernel is closely related to the ArcCosine
    kernel of order 0. The key reference is

    ::

        @incollection{NIPS2009_3628,
            title = {Kernel Methods for Deep Learning},
            author = {Youngmin Cho and Lawrence K. Saul},
            booktitle = {Advances in Neural Information Processing Systems 22},
            year = {2009},
            url = {http://papers.nips.cc/paper/3628-kernel-methods-for-deep-learning.pdf}
        }

    Note: broadcasting over leading dimensions has not yet been implemented for
    the ArcCosine kernel.
"""
mutable struct ArcCosine{T1,T2,T3,T4,T5,T6,T7,T8} <: Kernel
    input_dim::T1
    order::T2
    variance::T3
    weight_variances::T4
    bias_variance::T5
    active_dims::T6
    ARD::T7
    name::T8
    o::Union{PyObject, Nothing}
end

function ArcCosine(
    input_dim;
    order=0,
    variance=1.0,
    weight_variances=1.,
    bias_variance=1.,
    active_dims=nothing,
    ARD=nothing,
    name=nothing
)
    out = ArcCosine(
        input_dim,
        order,
        variance,
        weight_variances,
        bias_variance,
        active_dims,
        ARD,
        name,
        nothing
    )
    instantiate!(out)
    return out
end

function instantiate!(o::Union{ArcCosine, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.kernels.ArcCosine(
        o.input_dim,
        order=o.order,
        variance=o.variance,
        weight_variances=o.weight_variances,
        bias_variance=o.bias_variance,
        active_dims=o.active_dims,
        ARD=o.ARD,
        name=o.name
    )
    return o.o
end

"""
    The periodic family of kernels. Can be used to wrap any Stationary kernel
    to transform it into a periodic version. The canonical form (based on the
    SquaredExponential kernel) can be found in Equation (47) of

    D.J.C.MacKay. Introduction to Gaussian processes. In C.M.Bishop, editor,
    Neural Networks and Machine Learning, pages 133--165. Springer, 1998.

    The derivation can be achieved by mapping the original inputs through the
    transformation u = (cos(x), sin(x)).

    For the SquaredExponential base kernel, the result can be expressed as:
        k(r) = σ² exp{ -0.5 sin²(π r / γ) / ℓ² }

    where:
    r is the Euclidean distance between the input points,
    ℓ is the lengthscale parameter,
    σ² is the variance parameter,
    γ is the period parameter.

    (note that usually we have a factor of 4 instead of 0.5 in front but this
    is absorbed into lengthscale hyperparameter).
"""
mutable struct Periodic{T1,T2,T3,T4,T5,T6} <: Kernel
    input_dim::T1
    period::T2
    variance::T3
    lengthscales::T4
    active_dims::T5
    name::T6
    o::Union{PyObject, Nothing}
end

function Periodic(
    input_dim;
    period=1.0,
    variance=1.0,
    lengthscales=1.0,
    active_dims=nothing,
    name=nothing
)
    out = Periodic(
        input_dim,
        period,
        variance,
        lengthscales,
        active_dims,
        name,
        nothing
    )
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Periodic, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.kernels.Periodic(
        o.input_dim;
        period=o.period,
        variance=o.variance,
        lengthscales=o.lengthscales,
        active_dims=o.active_dims,
        name=o.name
    )
    return o.o
end

"""
    A Coregionalization kernel. The inputs to this kernel are _integers_
    (we cast them from floats as needed) which usually specify the
    *outputs* of a Coregionalization model.

    The parameters of this kernel, W, kappa, specify a positive-definite
    matrix B.

        B = W W^T + diag(kappa) .

    The kernel function is then an indexing of this matrix, so

        K(x, y) = B[x, y] .

    We refer to the size of B as "num_outputs x num_outputs", since this is
    the number of outputs in a coregionalization model. We refer to the
    number of columns on W as 'rank': it is the number of degrees of
    correlation between the outputs.

    NB. There is a symmetry between the elements of W, which creates a
    local minimum at W=0. To avoid this, it's recommended to initialize the
    optimization (or MCMC chain) using a random W.
"""
mutable struct Coregion{T1,T2,T3} <: Kernel
    output_dim::T1
    rank::T2
    active_dims::T3
    o::Union{PyObject, Nothing}
end

function Coregion(
    output_dim,
    rank;
    active_dims=nothing,
)
    out = Coregion(
        output_dim,
        rank,
        active_dims,
        nothing
    )
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Coregion, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.kernels.Coregion(
        o.output_dim,
        o.rank;
        active_dims=o.active_dims
    )
    return o.o
end

end # module
