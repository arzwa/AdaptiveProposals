using AdaptiveProposals, Distributions, Parameters
import Distributions: logpdf

# Get some made-up data.
data = rand(MixtureModel([Normal(0, 2.4), Exponential(1.6)]), 1000);
data[1:10]

# Define a likelihood model (it is the data generating model).
# We have three parameters: μ, σ and λ.
lhood(θ, data) = loglikelihood(MixtureModel([Normal(θ.μ, θ.σ), Exponential(θ.λ)]), data);

# Define some prior:
prior(θ) = logpdf(Normal(), θ.μ) + logpdf(Exponential(5), θ.σ) + logpdf(Exponential(2), θ.λ);

# ... and then the posterior up to a normalizing constant:
post(θ, data) = prior(θ) + lhood(θ, data);

# Here's and MCMC algorithm, using Metropolis-within-Gibbs
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

# Let's try it out. Define the proposal functions, one for each parameter.
proposals = (
    μ=AdaptiveProposal(), 
    σ=PositiveProposal(),  # σ ∈ ℝ+ 
    λ=PositiveProposal()   # λ ∈ ℝ+
)

# These are the proposal densities (kernels):
map(proposal->proposal.kernel, proposals)

# Do the MCMC:
out = mcmc(10000, (μ=1.0, σ=3.0, λ=0.2), data, proposals);

# Check whether it makes sense:
map(k->mean(map(x->getfield(x[1], k), out)), [:μ, :σ, :λ])

# The kernels have been adapting:
map(proposal->proposal.kernel, proposals)

