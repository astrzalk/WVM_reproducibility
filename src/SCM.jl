# Most code below is taken from https://github.com/richardkwo/InvariantCausal.jl, 
# specifically SEM.jl with minor modifications to update for Julia 1.0
using LinearAlgebra # For identity matrix, I
using Distributions

abstract type SCM end

struct LinAddSCM <: SCM  
   p ::Int64 # Number of variables
   B ::Matrix{Float64} # Weighted Adjacency Matrix of coefficients
   variances ::Vector{Float64} # Variances for each variables error term
   function LinAddSCM(B, variances)
       @assert length(variances) == size(B,1) == size(B,2)
       @assert all(variances .> 0)
       p = length(variances) 
       new(p, B, variances)
   end
end

struct NonLinAddSCM <: SCM
   p ::Int64 # Number of variables
   A ::Matrix{Float64} # Adjacency Matrix (signed so weights can only be {-1,0,1})
   fs::Vector{Function} # Vector of Non-linear Functions for each variable
   # variances ::Vector{Float64} # Variances for each variables error term; NOTE: will reintroduce after the first sim
   function NonLinAddSCM(A, fs)#, variances)
       @assert size(A,1) == size(A,2) == length(fs)# == length(variances) 
       # @assert all(variances .> 0)
       # p = length(variances)
       p = size(A,1)
       new(p, A, fs)#, variances)
   end
end

function Base.show(io::IO, scm::LinAddSCM)
    println(io, "LinAdd SCM with $(scm.p) variables.") 
    println(io, "σ² = $(scm.variances).")
end

"""
    simulate(scm; d::UnivariateDistribution)
    simulate(scm, [n]; d::UnivariateDistribution)
    simulate(scm, [do_variables, do_values], [shift, additive], [n]; d::UnivariateDistribution)

Simulate from a Linear Additive (LinAdd) scm `scm`. `n` is the sample size.
`d` is a keyword argument that specifies which distribution to sample from for the error terms.
do-interventions can be performed by specifying vectors of `do_variables` and `do_values`.
shift-interventions can be performed by setting `shift=true` with `shift_values = do_variables`
and `shift_values = do_values`.
shift-interventions can be either additive by letting `additive=true` or multiplicative `additive=false`.
"""
function simulate(scm::LinAddSCM, d::UnivariateDistribution)
    ϵ = rand(d, scm.p) .* sqrt.(scm.variances) # Scale variances to our chosen variances
    return (I - scm.B) \ ϵ
end

function simulate(scm::LinAddSCM, n::Int64; d::UnivariateDistribution=Normal(0,1))
    return vcat(map(i -> simulate(scm, d), 1:n)'...)
end

function simulate(scm::LinAddSCM, d::UnivariateDistribution, 
                  interventions::Tuple{Vector{Int64}, Vector{Float64}},
                  shift, additive)
    do_variables, do_values = interventions
    @assert length(do_variables) == length(do_values)
    p = scm.p
    ϵ = rand(d, p) .* sqrt.(scm.variances)
    if !shift # If shift intervention is false, perform an atomic intervention
        ϵ[do_variables] .= do_values
        B = copy(scm.B)
        B[do_variables, :] .= 0
    else
        shift_variables, shift_values = do_variables, do_values
        if additive 
            ϵ[shift_variables] .+= shift_values
        else
            ϵ[shift_variables] .*= shift_values
        end
        B = copy(scm.B)
    end
    return (I - B) \ ϵ
end

function simulate(scm::LinAddSCM, interventions::Tuple{Vector{Int64}, Vector{Float64}}, n::Int64;
                  d::UnivariateDistribution=Normal(0,1),
                  shift=false, additive=true)
    return vcat(map(i -> simulate(scm, d, interventions, shift, additive), 1:n)'...)
end

# simulate observationa data, but allowing to give the full vector of distributions for every noises separately in the SCM
function simulate(scm::LinAddSCM, noise_dist::Vector{<:UnivariateDistribution}, n::Int64)
    # check
    @assert length(noise_dist) == scm.p
    
    # interventions
    jointdist = product_distribution(noise_dist)
    B = copy(scm.B)
    
    # simulate
    ϵ = rand(jointdist, n) .* sqrt.(scm.variances)
    
    return ((I - B) \ ϵ)'
end 

function simulate(scm::NonLinAddSCM, noise_dist::Vector{<:UnivariateDistribution}, n, 
                  noise=nothing, sum_or_prod=:sum, top_order=collect(1:6))
    # check
    @assert length(noise_dist) == scm.p
    if sum_or_prod == :sum
        g = xs -> dropdims(sum(xs, dims=2), dims=2)
    else
        g = xs -> dropdims(prod(xs, dims=2), dims=2)
    end

    # simulate
    if isnothing(noise)
        jointdist = product_distribution(noise_dist)
        ϵ = rand(jointdist, n) #.* sqrt.(scm.variances) # p × n; will reintroduce this term after the first sim we do 
    else
        ϵ = noise
    end
    A = scm.A
    fs = scm.fs
    x = zeros(n, p)
    for j in top_order
        inds = findall(!iszero, A[:, j])
        if length(inds) == 0
            x[:,j] .= ϵ[j, :]
        else
            signs = A[inds,j]
            x[:, j] .= g(fs[j].(x[:, inds] .* signs')) .+ ϵ[j, :]
        end
    end
    return x
end



"""
        stochastic_inter_sim(scm::LinAddSCM, noise_dist::Vector{<:UnivariateDistribution}, inter_vect::Tuple{Vector{Int64},
                             Vector{<:UnivariateDistribution}}, n::Int64)

Simulate from an SCM with stochastic interventions, that is do(X:=N), with N indep. of the rest, for a set list of variables X.

* `scm`         : a linear Structural Causal Model
* `noise_dist`  : list of the distributions of the noises in observational setting (assumed to be indep., with mean = 0 and var = 1)
* `inter_vect`  : tuple of indexes of variables to intervene on, and their corresponding interventional distr. (~N)
* `n`           : nbr of samples
"""
function stochastic_inter_sim(scm::LinAddSCM, noise_dist::Vector{<:UnivariateDistribution}, inter_vect::Tuple{Vector{Int64},
                                Vector{<:UnivariateDistribution}}, n::Int64)
    # check
    inter_variables, inter_distr = inter_vect
    @assert length(noise_dist) == scm.p >= length(inter_variables) == length(inter_distr)
    
    # interventions
    dist_vect = copy(noise_dist)
    dist_vect[inter_variables] = inter_distr
    jointdist = product_distribution(dist_vect)
    B = copy(scm.B)
    B[inter_variables,:] .= 0
    
    # simulate
    ϵ = rand(jointdist, n) .* sqrt.(scm.variances)
    
    return ((I - B) \ ϵ)'
end 

"""
        stochastic_inter_sim(scm::NonLinAddSCM, noise_dist::Vector{<:UnivariateDistribution}, inter_vect::Tuple{Vector{Int64},
                             Vector{<:UnivariateDistribution}}, n::Int64, fs::Vector{Function})

Simulate from a non-linear SCM with stochastic interventions, that is do(X:=N), with N indep. of the rest, for a set list of variables X.

* `scm`         : a non-linear Structural Causal Model
* `noise_dist`  : list of the distributions of the noises in observational setting (assumed to be indep., with mean = 0 and var = 1)
* `inter_vect`  : tuple of indexes of variables to intervene on, and their corresponding interventional distr. (~N)
* `n`           : nbr of samples
* `fs`          : vector of non-linear functions used for the graph
"""
function stochastic_inter_sim(scm::NonLinAddSCM, noise_dist::Vector{<:UnivariateDistribution}, inter_vect, n::Int64, fs::Vector{Function})
    # interventions
    dist_vect = copy(noise_dist)
    inter_variables, inter_distr = inter_vect
    dist_vect[inter_variables] = inter_distr
    A = copy(scm.A)
    A[:,inter_variables] .= 0
    new_scm = NonLinAddSCM(A, fs)
    return simulate(new_scm, dist_vect, n)
end

"""
        mech_changes_sim(scm::LinAddSCM, noise_dist::Vector{<:UnivariateDistribution}, inter_vect::Tuple{Vector{Int64},
                             Vector{<:UnivariateDistribution}}, n::Int64)

Simulate from an SCM with mechanism changes, by either rescaling (i.e. X = β*X' + A*ϵ, with A random) or shifting the noise (i.e. X = β*X' + C+ϵ, with C random) or changing the structural equation's coefficients (i.e. X = β'*X' + ϵ) - or both coeff change + shift/rescale noise.

* `scm`         : a linear Structural Causal Model
* `noise_dist`  : list of the distributions of the noises in observational setting (assumed to be indep., with mean = 0 and var = 1)
* `noise_change`: tuple of indexes of variables for scale/shift noise change, and corresponding distr. (~A/C)
* `coeff_change`: tuple of indexes of variables for coeff. changes, and corresponding new coeffs (β')
* `n`           : nbr of samples
* `additive`    : whether we shift (=true) or rescale (=false) the intervened noise
"""
function mech_changes_sim(scm::LinAddSCM, noise_dist::Vector{<:UnivariateDistribution}, noise_change::Tuple{Vector{Int64},
                                Vector{<:UnivariateDistribution}}, coeff_change::Tuple{Vector{Int64}, Array{Float64,2}}, n::Int64;
                                additive = true)
    # check
    noise_change_variables, noise_change_distr::Array{UnivariateDistribution{Continuous},1} = noise_change
    coeff_variables, coeff_mat = coeff_change
    @assert size(coeff_mat)[1] == length(coeff_variables) <= size(coeff_mat)[2] == length(noise_dist) == scm.p >=                      length(noise_change_variables) == length(noise_change_distr)
    
    # interventions
    joint_change_distr = product_distribution(noise_change_distr)
    B = copy(scm.B)
    B[coeff_variables,:] = coeff_mat
    
    # simulate
    jointdist = product_distribution(noise_dist)
    ϵ = rand(jointdist, n) .* sqrt.(scm.variances)
    noise_mat = (additive ? zeros(scm.p, n) : ones(scm.p, n))
    noise_mat[noise_change_variables,:] = rand(joint_change_distr, n)
    
    return ((I - B) \ (additive ? ϵ .+ noise_mat : ϵ .* noise_mat))' 
    
end

"""
        mech_changes_sim(scm::NonLinAddSCM, noise_dist::Vector{<:UnivariateDistribution}, inter_vect::Tuple{Vector{Int64},
                             Vector{<:UnivariateDistribution}}, n::Int64)

Simulate from an SCM with mechanism changes, by either rescaling (i.e. X = f(X) + A*ϵ, with A random) or shifting the noise 
(i.e. X = f(X) + C+ϵ, with C random) or changing the structural equation's coefficients (i.e. X = β'*X' + ϵ) - or both coeff change + shift/rescale noise.

* `scm`         : a non-linear Structural Causal Model
* `noise_dist`  : list of the distributions of the noises in observational setting (assumed to be indep., with mean = 0 and var = 1)
* `noise_change`: tuple of indexes of variables for scale/shift noise change, and corresponding distr. (~A/C)
* `n`           : nbr of samples
* `additive`    : whether we shift (=true) or rescale (=false) the intervened noise
"""
function mech_changes_sim(scm::NonLinAddSCM, noise_dist::Vector{<:UnivariateDistribution}, noise_change::Tuple{Vector{Int64},
                                Vector{<:UnivariateDistribution}}, n::Int64;
                                additive = true)
    # check
    noise_change_variables, noise_change_distr::Array{UnivariateDistribution{Continuous},1} = noise_change
    @assert length(noise_dist) == scm.p >= length(noise_change_variables) == length(noise_change_distr)
    
    # interventions
    joint_change_distr = product_distribution(noise_change_distr)
    
    # simulate
    jointdist = product_distribution(noise_dist)
    ϵ = rand(jointdist, n) #.* sqrt.(scm.variances) # NOTE: pretend it doesnt have variances for now
    noise_mat = (additive ? zeros(scm.p, n) : ones(scm.p, n))
    noise_mat[noise_change_variables,:] = rand(joint_change_distr, n)
    # although it seems like noise_dist is getting used here, it is isn't and the noise distribution is being passed in.
    return simulate(scm, noise_dist, n, (additive ? ϵ .+ noise_mat : ϵ .* noise_mat))  
end


# get direct causes for i
function causes(scm::LinAddSCM, i::Int64)
    @assert 1 <= i <= scm.p
    return (1:scm.p)[scm.B[i, :].!=0]
end

# get markov blannket for i
function markov_blanket(scm::LinAddSCM, i::Int64)
    blanket = causes(scm, i)
    children = (1:scm.p)[scm.B[:,i].!=0]
    blanket = vcat(blanket, children)
    for j in children
        blanket = vcat(blanket,(1:scm.p)[scm.B[j, :].!=0])
    end
    return unique(blanket[blanket .!= i])
end

# function returning the structural equation coeff. for variable i
function causalcoeff(scm::LinAddSCM, i::Int64)
    @assert 1 <= i <= scm.p
    return (scm.B[i, 1:scm.p .!= i])
end

"""
    fixed_linear_SCM(p, k; [lb=-2, ub=2, var_min=0.5, var_max=2])

Generate a fixed-graph acyclic SCM with `p` variables with `edges` connecting them, and random coefficients.
* Edges should be of form [(child, parent), ...]
* `lb`, `ub`: coeff  ~ unif[`lb`, `ub`] with random sign
* `var_min`, `var_max`: var of error ~ unif[`var.min`, `var.max`]
"""
function fixed_linear_SCM(edges::Array{Tuple{Int64, Int64},1}, p::Int64; lb=-2, ub=2, var_min=0.5^2, var_max=2^2)
    B = zeros(p, p)
    for e in edges 
        B[e...] = (rand() * (ub - lb) + lb) * sign(randn())
    end
    variances = rand(p) * (var_max - var_min) .+ var_min
    return LinAddSCM(B, variances)
end

"""
    random_linear_SCM(p, k; [lb=-2, ub=2, var_min=0.5, var_max=2])

Generate a random-graph acyclic SCM with `p` variables and `k` average degree, and random coefficients.
* `lb`, `ub`: coeff  ~ unif[`lb`, `ub`] with random sign
* `var_min`, `var_max`: var of error ~ unif[`var.min`, `var.max`]
"""
function random_linear_SCM(p::Int64, k::Int64; lb=-2, ub=2, var_min=0.5^2, var_max=2^2)
    B = zeros(p, p)
    B[rand(p, p) .< k / (p-1)] .= 1
    B[UpperTriangular(B).!=0] .= 0
    m = sum(B.==1)
    B[B.==1] .= (rand(m) * (ub - lb) .+ lb) .* sign.(randn(m))
    err_var = rand(p) * (var_max - var_min) .+ var_min
    _order = sample(1:p, p, replace=false)
    B = B[_order, _order]
    return LinAddSCM(B, err_var)
end

"""
    random_noise_intervened_SCM(scm::LinAddSCM, [p_intervened=2, noise_multiplier_min=0.5, noise_multiplier_max=2., avoid=[],
                                                   prob_coeff_unchanged=2/3, lb=-2, ub=2])

Produce a new SCM based on original SCM by changing coefficients and noise variances.
* `p_intervened`: randomly choose `p_intervened` variables to intervene; will avoid those specified in `avoid`
* [`noise_multiplier_min`, `noise_multiplier_max`]: interval that noise multiplier is uniformly sampled from
* `prob.coeff.unchanged`: probability that coefficient is not changed
* `[lb, ub]`: if to change, coefficient is drawn uniformly from this interval with random sign

Return: `scm_new`, `intervened_variables`
"""
function random_noise_intervened_SCM(scm::LinAddSCM;
                                     p_intervened=2, noise_multiplier_min=0.5, noise_multiplier_max=2., avoid=[],
                                     prob_coeff_unchanged=2/3, lb=-2, ub=2)
    B = copy(scm.B)
    p = scm.p
    err_var = copy(scm.variances)
    vars = sample(setdiff(collect(1:p), avoid), p_intervened, replace=false)
    for i in vars
        noise_multiplier = rand() * (noise_multiplier_max - noise_multiplier_min) + noise_multiplier_min
        err_var[i] = err_var[i] * noise_multiplier
        if rand() > prob_coeff_unchanged
            _J = (1:p)[B[i, :].!=0]
            B[i, _J] .= rand(length(_J)) * (ub - lb) .+ lb
        end
    end
    return LinAddSCM(B, err_var), vars
end

"""
        high_dim_SCM(p_anterior::Int64, p_posterior::Int64, k::Int64, parents_nbr::Int64 ; 
                            lb=-2, ub=2, var_min=0.5, var_max=2, shuffle = false)

Generate a linear SCM, by allowing to fix the number of the direct causes (i.e. parents) of the target. 
Mainly used for high-dimensional simulations. The variables are returned in their causal order.

* `p_anterior`          : nbr of non-desc variables to the target (which is the (`p_anterior`+1)th variable)
* `p_posterior`         : nbr of variables non-asc variables to the target
* `k`                   : averaged degree of every other nodes
* `parents_nbr`         : nbr of direct causes of the target
* `lb`, `ub`            : coeff  ~ unif[`lb`, `ub`] with random sign
* `var_min`, `var_max`  : var of error ~ unif[`var.min`, `var.max`]
"""
function high_dim_SCM(p_anterior::Int64, p_posterior::Int64, k::Int64, parents_nbr::Int64 ; 
                      lb=-2, ub=2, var_min=0.5^2, var_max=2^2)
    # check
    @assert parents_nbr <= p_anterior
    
    # vars
    p = p_anterior + p_posterior + 1
    target_idx = p_anterior + 1
    
    # adjacency matrix
    B = zeros(p, p)
    draws = rand(p, p)
    draws[UpperTriangular(draws).> 0] .= 0
    B[draws .> 1 - min(k,p-1) / (p-1)] .= 1
    
    # target parents
    B[target_idx,sortperm(draws[target_idx,:])] = vcat(zeros(p - parents_nbr), ones(parents_nbr))
    m = sum(B.==1)
    
    # generate the coeff
    B[B.==1] .= (rand(m) * (ub - lb) .+ lb) .* rand([-1,1],m)
    
    # generate the variances
    err_var = rand(p) * (var_max - var_min) .+ var_min

    # Normalize X*β so that var(X*β) ∝ var(ϵ)
    B_inv = inv(I - B)
    for i in 2:p
        b²ᵢ = B_inv[i, :].^2
        vᵢ = sqrt(dot(b²ᵢ, err_var))
        B[i,:] ./= vᵢ
    end

    return LinAddSCM(B, err_var)
end
