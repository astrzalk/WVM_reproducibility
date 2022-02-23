include("SCM.jl")

# function to select the variables to intervene on.
"""
    inter_var_selection(scm::LinAddSCM, target::Int64, E::Int64, share::Float64; overlap = 0)

Select variables to intervene on in each interventional environment. The first environment being considered to be observational, it splits the variables in E-1 parts, corresponding to each interventional environment. Return array containing each of these E-1 groups of variables. It is possible to let the algo choose only `share`% of the variables to be intervened on in at least one environment. Also, `overlap` can be use to make two different environments to share common intervened variables. 

* `scm`    : linear structural causal model
* `target` : index of target variable
* `E`      : nbr of environments (including observatinal one, hence E > 1)
* `share`  : share of the variables to get intervened on in at least one environment 
* `overlap`: every two concecutive env have in common `overlap`% of the intervened variables
"""
function inter_var_selection(scm::SCM, target::Int64, E::Int64, share::Float64; overlap = 0.2)
    # init
    @assert E > 1 <= target <= scm.p && 0 <= share <= 1 >= overlap >= 0
    E₀ = E - 1 # nbr of interventional envs
    p::Int64 = floor(share*(scm.p - 1)) # nbr of vars to intervene on, -1 is for the target
    vars = deleteat!(Vector(1:scm.p), target)
    inter_variables::Array{Array{Int64,1},1} = []
    
    # generic fct, test if var is intervened on
    test_inter(x, l, u) = (l * p <= x-1 < u * p) || (l * p <= x-1 - p) || (x-1 + p < u * p)
    
    # random selection
    vars = vars[sortperm(rand(scm.p-1))[1:p]]
    cst = 1 / E₀
    for e in 1:E₀
        l = (e-1) * cst - overlap / 2
        u = e * cst + overlap / 2
        push!(inter_variables, vars[filter(x -> test_inter(x, l, u), 1:p)])
    end
    
    return inter_variables
end

# Choose which variables to invene on in each environment with a simple hack:
# With probability 1/3 intervene on 1 variable, 1/3 intervene on 3 variables, 
# 1/3 intervene on 5 variables.
function simple_var_selection(scm::SCM, target::Int64, to_intervene = [1,3,5])
    @assert E > 1 <= target <= scm.p
    E₀ = E - 1 # nbr of interventional envs
    p = scm.p - 1 # nbr of vars to intervene on, -1 is for the target
    vars = deleteat!(Vector(1:scm.p), target)
    inter_variables::Array{Array{Int64,1},1} = []
    for e in 1:E₀
        # with probability 1/3 select either 1, 3, or 5 variables to intervene in env. e
        num_intervene = to_intervene[rand(Categorical([1/3,1/3,1/3]))] 
        push!(inter_variables, sample(vars, num_intervene, replace=false))
    end
    return inter_variables
end

"""
    inter_distr_selection(inter_variables::Array{Array{Int64,1},1} ; lb=-2, ub=2, var_min=0.5, var_max=2)

Generate the noise distributions (~N) for the stochastic interventions. Each of these distr. is Normal(μ,σ), and we randomly select μ and σ. Return a list of tuples (one per interventional environments) of the same type as `inter_vect` in `stochastic_inter_sim`.

* `inter_variables`   : output of `inter_var_selection`, i.e. array of arrays containing the variables to intervene on in each env.
* `lb`                : lower bound for unif distr. generating a μ (actually we also randomly flip the sign of μ after)
* `ub`                : upper bound for unif distr. generating a μ 
* `var_min`           : lower bound for unif distr. generating a σ^2
* `var_max`           : upper bound for unif distr. generating a σ^2
"""
function inter_distr_selection(inter_variables::Array{Array{Int64,1},1} ;
                                    lb=-2, ub=2, var_min=0.5^2, var_max=2^2)
    # init
    @assert 0 <= var_min <= var_max
    E = length(inter_variables)
    nSim = sum(length.(inter_variables))
    inter_distr::Array{Array{T,1},1} where T <: UnivariateDistribution = []
    
    # draw params
    means = (rand(nSim) .* (ub - lb) .+ lb) .* rand([-1,1], nSim)
    vars = rand(nSim) .* (var_max - var_min) .+ var_min
    
    # construct array of distr
    idx = 1
    for e in 1:E
        inter_dist_sub::Array{T,1} where T <:UnivariateDistribution = []
        for var in inter_variables[e]
            push!(inter_dist_sub, Normal(means[idx], sqrt(vars[idx])))
            idx = idx + 1
        end
        push!(inter_distr, inter_dist_sub)
    end
    
    return [tuple_ for tuple_ in zip(inter_variables, inter_distr)] 
end




"""
    inter_noise_selection(inter_variables::Array{Array{Int64,1},1} ;
                                    additive = true, prob_cst = 1/3, lb = 0, ub = 1)

Generate the distributions of the shift/rescale variables (i.e. A or C) for mechanism changes. Each of these distr. is either uniform or constant. Return a list of tuples (one per interventional environments) of the same type as `noise_change` in `mech_changes_sim`.


* `inter_variables`   : output of `inter_var_selection`, i.e. array of arrays containing the variables to intervene on in each env.
* `additive`          : whether the noise is shifted (true) or rescaled (false)
* `prob_cst`          : whether the shift/rescale variables are constant or not
* `lb`                : lower bound of interval where the bounds for the uniform distr. are randomly chosen (uniformly)
* `ub`                : upper bound... Note: if additive, we randomly flip the interval around zero, else we flip around 1
* `flip`              : whether we randomly flip the intervals as mentionned above
"""
function inter_noise_selection(inter_variables::Array{Array{Int64,1},1} ;
                                    additive = true, prob_cst = 1/3, lb = 0, ub = 1, flip = true)
    # init
    @assert 0 <= prob_cst <= 1 && !(!additive && flip) || 0 < lb <= ub
    E = length(inter_variables)
    nSim = sum(length.(inter_variables))
    inter_distr::Array{Array{UnivariateDistribution,1},1} = []
    
    # draw params
    bds_1 = rand(nSim) .* (ub - lb) .+ lb
    bds_2 = rand(nSim) .* (ub - lb) .+ lb
    cst_probs = rand(nSim)
    switches = flip ? rand([0,1], nSim) : zeros(nSim)
    lower_bds = min.(bds_1, bds_2) .+ switches .* (.- min.(bds_1, bds_2) .+ (additive ? .- max.(bds_1, bds_2) : max.(bds_1, bds_2).^(-1)))
    upper_bds = max.(bds_1, bds_2) .+ switches .* (.- max.(bds_1, bds_2) .+ (additive ? .- min.(bds_1, bds_2) : min.(bds_1, bds_2).^(-1)))
    
    # construct array of distr
    idx = 1
    for e in 1:E
        inter_dist_sub::Array{UnivariateDistribution ,1} = []
        for var in inter_variables[e]
            noise_distr = Normal(0,1)
            if cst_probs[idx] <= prob_cst
                noise_distr = Normal((lower_bds[idx] + upper_bds[idx])/2,0)
            else
                noise_distr = Uniform(lower_bds[idx], upper_bds[idx])
            end
            push!(inter_dist_sub, noise_distr)
            idx = idx + 1
        end
        push!(inter_distr, inter_dist_sub)
    end
    
    return [tuple_ for tuple_ in zip(inter_variables, inter_distr)] 
end

function simple_noise_selection(inter_variables::Array{Array{Int64,1},1}, d=TDist(4))
    meanshift = [0,0.1,0.2,0.5,1,2,5,10]
    strength = [0.1,0.2,0.5,1,2,5,10]
    E = length(inter_variables)
    inter_distr = Vector{Vector{<:LocationScale{Float64, <:UnivariateDistribution}}}(undef, E) #TDist{Float64}
    for e in 1:E
        num_int = length(inter_variables[e])
        tmp_dist::LocationScale{Float64, <:UnivariateDistribution} = rand(strength)*d + rand(meanshift)
        inter_dist_sub::Vector{<:LocationScale{Float64, <:UnivariateDistribution}} = [tmp_dist]#(undef, num_int) # TDist{Float64}
        for i in 2:num_int
#             inter_dist_sub[i] = rand(strength)*d + rand(meanshift)
            push!(inter_dist_sub, rand(strength)*d + rand(meanshift))
        end
        inter_distr[e] =  inter_dist_sub
    end
    return [tuple_ for tuple_ in zip(inter_variables, inter_distr)] 
end

"""
    inter_coeff_selection(scm::LinAddSCM, inter_variables::Array{Array{Int64,1},1} ;
                                prob_unchanged = 2/3, redraw = false, lb=-2, ub=2, variance = 1)

Generate the new coeefficents from mechanism changes. It eithers generate completely new coefficent from scratch or just add some 'noise' to the existing coeff (this 'noise' is here constant when generating samples). With a certain probability it will not change the coeff. Return a list of tuples (one per interventional environments) of the same type as `coeff_change` in `mech_changes_sim`.

* `scm`               : the linear structural model
* `inter_variables`   : output of `inter_var_selection`, i.e. array of arrays containing the variables to intervene on in each env.
* `prob_unchanged`    : probability that a coeff is eventually unchanged
* `redraw`            : whether we redraw totally new coeff (true) or change the existing coeff by adding some noise (false)
* `lb`                : lower bound of uniform distr used to draw new coeff (when redraw = true). We also randomly flip signs after.
* `ub`                : upper bound of uniform distr used to draw new coeff.
* `variance`          : the noise term added to the coeff when redraw = false is drawn from N(0, `variance` )
"""
function inter_coeff_selection(scm::LinAddSCM, inter_variables::Array{Array{Int64,1},1} ;
                                prob_unchanged = 2/3, redraw = false, lb=-2, ub=2, variance = 1)
    @assert 0 <= prob_unchanged <= 1
    E = length(inter_variables)
    nSim = sum(map(x -> sum(scm.B[x,:] .!= 0), inter_variables))
    coeff_mat::Array{Array{Float64, 2}}= []
    
    # draw params
    unchange_probs = rand(nSim)
    coeff_changes = redraw ? (rand(nSim) .* (ub - lb) .+ lb) .* rand([-1,1],nSim) : rand(Normal(0,variance), nSim)
    
    # construct array of distr
    idx = 1
    for e in 1:E
        coeff_mat_sub::Array{Float64, 2} = zeros(0,scm.p)
        for var in inter_variables[e]
            vect = scm.B[var, :]
            for j in filter(i -> vect[i] != 0, 1:scm.p)
                if unchange_probs[idx] > prob_unchanged
                    vect[j] = redraw ? coeff_changes[idx] : vect[j] + coeff_changes[idx] 
                end
                idx = idx +1
            end
            coeff_mat_sub = vcat(coeff_mat_sub, vect')
        end
        push!(coeff_mat, coeff_mat_sub)
    end
    
    return [tuple_ for tuple_ in zip(inter_variables, coeff_mat)] 
end

# struct for linear structural models together with all its modifications under different interventional evironments and the information needed to generate samples from under both observational and interventional settings. 
struct InterLinAddSCM 
    scm::LinAddSCM
    is_mech_change::Bool
    inter_vect_list::Array{Tuple{Vector{Int64}, Vector{UnivariateDistribution }},1} # for stochastic_inter_sim
    noise_change_list::Array{Tuple{Vector{Int64}, Vector{UnivariateDistribution }},1}  # for mech_changes_sim
    coeff_change_list::Array{Tuple{Array{Int64,1},Array{Float64,2}},1} # for mech_changes_sim
    E # number of environments
    target # target variable
    additive # when the intervention is a mechanism change, if true -> shifted noise, otherwise rescaled by a random variable
    function InterLinAddSCM(scm::LinAddSCM, is_mech_change::Bool, target::Int64; additive = true,
                                inter_vect_list =[], #::Array{<:Tuple{Vector{Int64}, Vector{<:UnivariateDistribution}},1} = [],
                                noise_change_list = [],#::Array{<:Tuple{Vector{Int64}, Vector{<:UnivariateDistribution}},1} = [],
                                coeff_change_list = [])#::Array{<:Tuple{Array{Int64,1}, Array{Float64,2}},1} = [])
        @assert (!is_mech_change || length(noise_change_list) == length(coeff_change_list)) && 1 <= target <= scm.p
        E = is_mech_change ? length(noise_change_list) + 1 : length(inter_vect_list) + 1
        new(scm, is_mech_change, inter_vect_list, noise_change_list, coeff_change_list, E, target,additive)
    end
end

# struct for non-linear structural models together with all its modifications under different interventional evironments and the information needed to generate samples from under both observational and interventional settings. 
struct InterNonLinAddSCM 
    scm::NonLinAddSCM
    is_mech_change::Bool
    inter_vect_list#::Array{Tuple{Vector{Int64}, Vector{T}},1} where T <:LocationScale{Float64, TDist{Float64}}# for stochastic_inter_sim
    noise_change_list#::Array{Tuple{Vector{Int64}, Vector{T}},1} where T <:LocationScale{Float64, TDist{Float64}}# for mech_changes_sim
    E # number of environments
    target # target variable
    additive # when the intervention is a mechanism change, if true -> shifted noise, otherwise rescaled by a random variable
    function InterNonLinAddSCM(scm::NonLinAddSCM, is_mech_change::Bool, target::Int64; additive = true,
                                inter_vect_list =[], #::Array{<:Tuple{Vector{Int64}, Vector{<:UnivariateDistribution}},1} = [],
                                noise_change_list = [])
        @assert 1 <= target <= scm.p
        E = is_mech_change ? length(noise_change_list) + 1 : length(inter_vect_list) + 1
        new(scm, is_mech_change, inter_vect_list, noise_change_list, E, target,additive)
    end
end



"""
    generate_interLinAddSCM(is_mech_change::Bool, E::Int64, share::Float64; scm = nothing,
        high_dim_SCM_params = Dict("p_anterior" => 50, "p_posterior" => 50, "k" => 20, "parents_nbr" => 20, "lb" => -2, "ub" => 2, "var_min"    => 0.5^2, "var_max" => 2^2), overlap = 0.2, additive = true, inter_noise_selection_params = Dict("prob_cst" => 1/3, "lb" => 0, "ub" => 1),
        inter_coeff_selection_params = Dict("prob_unchanged" => 2/3, "redraw" => false, "lb" => -2, "ub" => 2, "variance" => 1),
        inter_distr_selection_params = Dict("lb" => -2, "ub" => 2, "var_min" => 0.5^2, "var_max" => 2^2))

Generate from scratch a new linear structural model and different interventional environment from it. Returns an InterLinAddSCM object. Interventional environments can only be either fully made of stochastic interventions or fully made of mechanism changes. The first environment is always considered to be purely observational.

* `is_mech_change`                : whether the interventions are mechanism changes (true) or not (false)
* `E`                             : number of environments, E >= 1
* `share`                         : between 0 and 1, share of the non-target variables that are intervened on at least once
* `scm`                           : can provide an SCM instead of generating one (generate one if `scm` = nothing)
* `target`                        : target variable (only when providing an SCM in `scm` field)
* `high_dim_SCM_params`           : parameters for the function `high_dim_SCM`
* `overlap`                       : between 0 and 1, overlap of inter. variables between two env, see `inter_var_selection`
* `additive`                      : for mech change, whether the noise change is additive (true) or multiplicative (false)
* `inter_noise_selection_params`  : parameters for the function `inter_noise_selection`
* `inter_coeff_selection_params`  : parameters for the function `inter_coeff_selection`
* `inter_distr_selection_params`  : parameters for the function `inter_distr_selection`
"""
function generate_interLinAddSCM(is_mech_change::Bool, E::Int64, share::Float64; scm = nothing, target = -1,
        high_dim_SCM_params = Dict("p_anterior" => 50, "p_posterior" => 50, "k" => 20, "parents_nbr" => 20, "lb" => -2, "ub" => 2, 
        "var_min" => 0.5^2, "var_max" => 2^2), overlap = 0.2, additive = true, 
        inter_noise_selection_params = Dict("prob_cst" => 1/3, "lb" => 0, "ub" => 1, "flip" => true),  
        inter_coeff_selection_params = Dict("prob_unchanged" => 2/3, "redraw" => false, "lb" => -2, "ub" => 2, "variance" => 1),
        inter_distr_selection_params = Dict("lb" => -2, "ub" => 2, "var_min" => 0.5^2, "var_max" => 2^2))
    # check
    @assert E >= 1 && 0 <= share <= 1
        
    # create the linear SCM if it is not already given
    if scm == nothing
        hdsp = high_dim_SCM_params
        p_anterior, p_posterior, k, parents_nbr = hdsp["p_anterior"], hdsp["p_posterior"], hdsp["k"], hdsp["parents_nbr"]
        lb, ub, var_min, var_max = hdsp["lb"], hdsp["ub"], hdsp["var_min"], hdsp["var_max"]
        scm = high_dim_SCM(p_anterior,p_posterior,k,parents_nbr; lb=lb, ub=ub, var_min=var_min, var_max=var_max)
        target = p_anterior + 1
    else
        @assert typeof(scm) == LinAddSCM && 1 <= target <= scm.p
    end
    
    # create the interventional envs if needed
    if E > 1
        # generate variables to intervene on by (interventional) environments
        inter_variables = inter_var_selection(scm, target, E, share; overlap = overlap)
        
        # generate now the params for each interventional env, and return a InterLinAddSCM
        if is_mech_change
            # generate the noise scaling/shifting variables per intervention
            insp = inter_noise_selection_params
            prob_cst, lb, ub, flip = insp["prob_cst"], insp["lb"], insp["ub"], insp["flip"]
            noise_change_list = inter_noise_selection(inter_variables; 
                                    additive = additive, prob_cst = prob_cst, lb = lb, ub = ub, flip = flip)
            
            # generate the potential coefficient changes per intervention
            icsp = inter_coeff_selection_params
            prob_unchanged, redraw, lb, ub, variance = icsp["prob_unchanged"], icsp["redraw"], icsp["lb"], icsp["ub"], icsp["variance"]
            coeff_change_list = inter_coeff_selection(scm, inter_variables; 
                                    prob_unchanged = prob_unchanged, redraw = redraw, lb=lb, ub=lb, variance = variance)
            
            return InterLinAddSCM(scm,is_mech_change,target; 
                                    additive = additive, noise_change_list = noise_change_list, coeff_change_list = coeff_change_list)
        else
            # generate the do-intervention noise distributions
            idsp = inter_distr_selection_params
            lb, ub, var_min, var_max = idsp["lb"], idsp["ub"], idsp["var_min"], idsp["var_max"]
            inter_vect_list = inter_distr_selection(inter_variables; lb=lb, ub=ub, var_min=var_min, var_max=var_max)
            
            return InterLinAddSCM(scm,is_mech_change,target; additive = additive, inter_vect_list = inter_vect_list)
        end
    else
        return InterLinAddSCM(scm,is_mech_change,target)
    end
end

"""
    simulate(inter_scm::InterLinAddSCM, nₑs, d::UnivariateDistribution)

Simulate observational and interventional data from an interventional linear SCM object, i.e. InterLinAddSCM. 

* `inter_scm`   : the InterLinAddSCM object
* `nₑs`         : array of the number of samples from environment
* `ds`          : vector of (observational) distributions of the noises, assumed to be of mean = 0 and variance = 1, e.g. ds=[Gamma(), Exponential(), ...]
"""
function simulate(inter_scm::InterLinAddSCM, nₑs, ds::Vector{T}) where T <: UnivariateDistribution
    # check & init
    E = inter_scm.E
    @assert length(nₑs) == E
    p = inter_scm.scm.p
    target = inter_scm.target
    is_mech_change = inter_scm.is_mech_change
    scm = inter_scm.scm
    additive = inter_scm.additive
    
    # simulate data
    obs_noise_distr = rand(ds, p)

    # observational data
    sim_data = simulate(scm, obs_noise_distr, nₑs[1])
    
    # interventional data
    for e in 1:(E-1)
        if is_mech_change 
            noise_change = inter_scm.noise_change_list[e]
            coeff_change = inter_scm.coeff_change_list[e]
            sim_data = vcat(sim_data, mech_changes_sim(scm, obs_noise_distr, noise_change, coeff_change, nₑs[e+1]; additive = additive))
        else
            inter_vect = inter_scm.inter_vect_list[e]
            sim_data = vcat(sim_data, stochastic_inter_sim(scm, obs_noise_distr, inter_vect, nₑs[e+1]))
        end
    end
    
    # separate target (y) from the rest (X)
    X, y = sim_data[:,1:p .!= target], sim_data[:,target]
        
    return (X,y)
end


"""
    simulate(inter_scm::InterNonLinAddSCM, nₑs, d::UnivariateDistribution)

Simulate observational and interventional data from an interventional non-linear SCM object, i.e. InterNonLinAddSCM. 

* `inter_scm`   : the InterNonLinAddSCM object
* `nₑs`         : array of the number of samples from environment
* `ds`          : vector of (observational) distributions of the noises, assumed to be of mean = 0 and variance = 1, e.g. ds=[Gamma(), Exponential(), ...]
"""
function simulate(inter_scm::InterNonLinAddSCM, nₑs, ds::Vector{T}) where T <: UnivariateDistribution
    # check & init
    E = inter_scm.E
    @assert length(nₑs) == E
    p = inter_scm.scm.p
    target = inter_scm.target
    is_mech_change = inter_scm.is_mech_change
    scm = inter_scm.scm
    additive = inter_scm.additive
    
    # simulate data
    obs_noise_distr = rand(ds, p)

    # observational data
    sim_data = simulate(scm, obs_noise_distr, nₑs[1])
    
    # interventional data
    for e in 1:(E-1)
        if is_mech_change 
            noise_change = inter_scm.noise_change_list[e]
            sim_data = vcat(sim_data, mech_changes_sim(scm, obs_noise_distr, noise_change, nₑs[e+1]; additive = additive))
        else
            inter_vect = inter_scm.inter_vect_list[e]
            sim_data = vcat(sim_data, stochastic_inter_sim(scm, obs_noise_distr, inter_vect, nₑs[e+1], scm.fs))
        end
    end
    
    # separate target (y) from the rest (X)
    X, y = sim_data[:,1:p .!= target], sim_data[:,target]
        
    return (X,y)
end
