abstract type Kernel end


import Base: +, *, zero, cos
using Distances: sqeuclidean, SqEuclidean, Euclidean
using Base.Broadcast: broadcast_shape
using LinearAlgebra: isposdef, checksquare

const AV{T} = AbstractVector{T}
const AM{T} = AbstractMatrix{T}
const AVM{T} = AbstractVecOrMat{T}

export
    Kernel,
    ZeroKernel,
    zero
abstract type Kernel end



"""
    ZeroKernel <: Kernel
A rank 0 `Kernel` that always returns zero.
"""
struct ZeroKernel{T<:Real} <: Kernel end
ZeroKernel() = ZeroKernel{Float64}()
zero(::Kernel) = ZeroKernel()

function (k::ZeroKernel{T})(x::AV, x′::AV; type=:pairwise) where {T}
    if type == :elementwise
        return zeros(T, broadcast_shape(size(x), size(x′)))
    elseif type == :pairwise
        return zeros(T, length(x), length(x′))
    else
        @error "`$type` is not a valid kernel application type" 
    end
end
