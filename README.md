```@meta
EditURL = "README.jl"
```

````julia
using AdaptiveProposals, Distributions, Parameters
import Distributions: logpdf
````

Get some made-up data.

````julia
data = rand(MixtureModel([Normal(0, 2.4), Exponential(1.6)]), 1000);
data[1:10]
````

````
10-element Vector{Float64}:
  0.324469051967057
  2.816562623155402
 -2.23041581032082
 -2.9675288011196974
  0.17830696630396375
 -5.537938077992835
  0.31817677428273505
  0.740301548669381
  2.202501105492761
  4.377902039469137
````

Define a likelihood model (it is the data generating model).
We have three parameters: μ, σ and λ.

````julia
lhood(θ, data) = loglikelihood(MixtureModel([Normal(θ.μ, θ.σ), Exponential(θ.λ)]), data);
````

Define some prior:

````julia
prior(θ) = logpdf(Normal(), θ.μ) + logpdf(Exponential(5), θ.σ) + logpdf(Exponential(2), θ.λ);
````

... and then the posterior up to a normalizing constant:

````julia
post(θ, data) = prior(θ) + lhood(θ, data);
````

Here's and MCMC algorithm, using Metropolis-within-Gibbs

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
end;
````

Let's try it out. Define the proposal functions, one for each parameter.

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

These are the proposal densities (kernels):

````julia
map(proposal->proposal.kernel, proposals)
````

````
(μ = Normal{Float64}(μ=0.0, σ=1.0), σ = Normal{Float64}(μ=0.0, σ=1.0), λ = Normal{Float64}(μ=0.0, σ=1.0))
````

Do the MCMC:

````julia
out = mcmc(10000, (μ=1.0, σ=3.0, λ=0.2), data, proposals);
````

Check whether it makes sense:

````julia
map(k->mean(map(x->getfield(x[1], k), out)), [:μ, :σ, :λ])
````

````
3-element Vector{Float64}:
 0.1395503626963868
 2.5431243656053755
 1.462259812138308
````

The kernels have been adapting:

````julia
map(proposal->proposal.kernel, proposals)
````

````
(μ = Normal{Float64}(μ=0.0, σ=0.31809099463761914), σ = Normal{Float64}(μ=0.0, σ=0.13489077762567608), λ = Normal{Float64}(μ=0.0, σ=0.13383978521186746))
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

