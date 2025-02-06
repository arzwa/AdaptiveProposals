```@meta
EditURL = "README.jl"
```

````julia
using AdaptiveProposals, Distributions, Parameters
import Distributions: logpdf
````

A made-up data generating model

````julia
data = rand(MixtureModel([Normal(0, 2.4), Exponential(1.6)]), 1000);
data[1:10]
````

````
10-element Vector{Float64}:
  0.12816376552104622
 -2.490551654111598
 -7.296960665986657
  0.41401835767665
 -0.5901497122609962
  0.23898197008583347
  0.12008970277615286
  0.47895130269355957
 -0.316291856158849
  0.5983797389277226
````

Likelihood

````julia
lhood(θ, data) = loglikelihood(MixtureModel([Normal(θ.μ, θ.σ), Exponential(θ.λ)]), data)
````

````
lhood (generic function with 1 method)
````

Prior

````julia
prior(θ) = logpdf(Normal(), θ.μ) + logpdf(Exponential(5), θ.σ) + logpdf(Exponential(2), θ.λ)
````

````
prior (generic function with 1 method)
````

Posterior up to a normalizing constant

````julia
post(θ, data) = prior(θ) + lhood(θ, data)
````

````
post (generic function with 1 method)
````

MCMC function, using Metropolis-within-Gibbs

````julia
function mcmc(n, θ, data, proposals)
    π  = post(θ, data)
    x  = (θ, π)
    xs = Vector{typeof(x)}(undef, n)
    for i=1:n
        for k=[:μ, :σ, :λ]  # Gibbs updates, update param x | param y, z, ...
            prop = getfield(proposals, k)
            θk = getfield(θ, k)
            θk_, q = prop(θk)    # new proposed parameter value and log proposal density ratio
            θ_ = reconstruct(θ, k=>θk_)
            π_ = post(θ_, data)
            ar = π_ - π + q      # calculate the log acceptance probabilty
            if log(rand()) < ar  # accept/reject
                θ = θ_
                π = π_
                accept!(prop)    # notify acceptance to adapt the proposal kernel
            end
        end
        xs[i] = (θ, π)
    end
    return xs
end
````

````
mcmc (generic function with 1 method)
````

Try it out

````julia
proposals = (
    μ=AdaptiveProposal(),
    σ=PositiveProposal(),  # σ ∈ ℝ+
    λ=PositiveProposal()   # λ ∈ ℝ+
)
````

````
(μ = AdaptiveProposal{Normal{Float64}, typeof(identity), typeof(identity)}
  kernel: Normal{Float64}
  trans: identity (function of type typeof(identity))
  invtrans: identity (function of type typeof(identity))
  tuneinterval: Int64 25
  total: Int64 0
  accepted: Int64 0
  δmax: Float64 0.2
  logbound: Float64 10.0
  target: Float64 0.36
  stop: Int64 100000
, σ = AdaptiveProposal{Normal{Float64}, typeof(log), typeof(exp)}
  kernel: Normal{Float64}
  trans: log (function of type typeof(log))
  invtrans: exp (function of type typeof(exp))
  tuneinterval: Int64 25
  total: Int64 0
  accepted: Int64 0
  δmax: Float64 0.2
  logbound: Float64 10.0
  target: Float64 0.36
  stop: Int64 100000
, λ = AdaptiveProposal{Normal{Float64}, typeof(log), typeof(exp)}
  kernel: Normal{Float64}
  trans: log (function of type typeof(log))
  invtrans: exp (function of type typeof(exp))
  tuneinterval: Int64 25
  total: Int64 0
  accepted: Int64 0
  δmax: Float64 0.2
  logbound: Float64 10.0
  target: Float64 0.36
  stop: Int64 100000
)
````

These are the proposal kernels:

````julia
map(proposal->proposal.kernel, proposals)
````

````
(μ = Normal{Float64}(μ=0.0, σ=1.0), σ = Normal{Float64}(μ=0.0, σ=1.0), λ = Normal{Float64}(μ=0.0, σ=1.0))
````

Do the MCMC

````julia
out = mcmc(10000, (μ=1.0, σ=3.0, λ=0.2), data, proposals);
````

Check whether it makes sense:

````julia
map(k->mean(map(x->getfield(x[1], k), out)), [:μ, :σ, :λ])
````

````
3-element Vector{Float64}:
 -0.29506293930177
  2.3599536132454513
  1.7065432577084365
````

The kernels have been adapting:

````julia
map(proposal->proposal.kernel, proposals)
````

````
(μ = Normal{Float64}(μ=0.0, σ=0.41443469974779795), σ = Normal{Float64}(μ=0.0, σ=0.14076051803250902), λ = Normal{Float64}(μ=0.0, σ=0.1350493588505622))
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

