module AdaptiveProposals

using Bijectors, Distributions, Parameters, Random
export AdaptiveProposal, UnitIntervalProposal, PositiveProposal, accept!

# a default adaptive unit interval proposal
UnitIntervalProposal(;kwargs...) = AdaptiveProposal(trans=Bijectors.Logit(0,1); kwargs...)
PositiveProposal(;kwargs...) = AdaptiveProposal(trans=log; kwargs...)

"""
    AdaptiveProposal(kernel)

An adaptive Univariate proposal kernel, assumes the support is ℝ.
"""
@with_kw mutable struct AdaptiveProposal{T,V,W} 
    kernel      ::T       = Normal()
    trans       ::V       = identity()
    invtrans    ::W       = inverse(trans)
    tuneinterval::Int64   = 25
    total       ::Int64   = 0
    accepted    ::Int64   = 0
    δmax        ::Float64 = 0.2
    logbound    ::Float64 = 10.
    target      ::Float64 = 0.36
    stop        ::Int64   = 100000
end

const Symmetric = Union{Normal,Uniform}

function (prop::AdaptiveProposal)(θ)
    consider_adaptation!(prop)
    prop.total += 1
    return propose(prop, θ)
end

propose(k::AdaptiveProposal, x::Float64) = propose(Random.default_rng(), k, x)
function propose(rng::AbstractRNG, k::AdaptiveProposal{T}, x::Float64) where T<:Symmetric
    @unpack trans, invtrans, kernel = k
    y  = transform(trans, x)
    y′ = y + rand(rng, kernel)
    x′ = transform(invtrans, y′) 
    return x′, logabsdetjac(trans, x) - logabsdetjac(trans, x′)
end

function adapt!(x::AdaptiveProposal{T}) where T<:Distribution
    @unpack total, tuneinterval, accepted, δmax, target, logbound = x
    (total == 0) && return
    δn = min(δmax, 1. /√(total/tuneinterval))
    α  = accepted / tuneinterval
    lσ = α > target ? log(hyperp(x)) + δn : log(hyperp(x)) - δn
    lσ = abs(lσ) > logbound ? sign(lσ) * logbound : lσ
    x.kernel = adapted(x.kernel, lσ)
    x.accepted = 0
end

function consider_adaptation!(prop)
    a = prop.total <= prop.stop 
    b = prop.total % prop.tuneinterval == 0
    (a && b) && adapt!(prop)
end

accept!(p::AdaptiveProposal) = p.accepted += 1

hyperp(x::AdaptiveProposal{Uniform{T}}) where T<:Real = x.kernel.b
hyperp(x::AdaptiveProposal{Normal{T}})  where T<:Real = x.kernel.σ

adapted(_::Uniform, lσ::Float64) = Uniform(-exp(lσ), exp(lσ))
adapted(_::Normal,  lσ::Float64) = Normal(0., exp(lσ))


end # module AdaptiveProposals
