using LinearAlgebra
using Lasso
using Distributions
using ProximalOperators
using Optim
using Distances
using Statistics
using DataFrames # for Not

# direct one dimensional Wasserstein variance
mutable struct DirectWassersteinVariance
    X
    y
    nₑs
    w
    E
    E_inds
    Δπ_vect
    idx
    function DirectWassersteinVariance(X, y, nₑs, w)
        # check
        @assert size(X,1) == size(y,1) == sum(nₑs)
        @assert abs(sum(w)-1) <= 1e-10
        E = length(nₑs)
        
        # get data per environment
        E_inds = vcat([0],cumsum(nₑs))
        
        # probability masses for the barycenter
        temp_vect = zeros(E_inds[E+1])
        for e in 1:E
            temp_vect[(E_inds[e] + 1) : E_inds[e+1]] = (1:nₑs[e]) ./ nₑs[e]
        end
        π_vect = sort(unique(temp_vect))
        n = length(π_vect)
        Δπ_vect = vcat(π_vect[1], π_vect[2:n] - π_vect[1:(n-1)])
        
        # indices corresponding to masses of barycenter, if data sorted in each env
        idx::Array{Int64,2} = zeros(n,E)
        for e in 1:E
            idx[:,e] .= ceil.(π_vect .* nₑs[e])
        end
        
        # create object
        new(X,y,nₑs,w,E,E_inds,Δπ_vect,idx) 
    end
end

# compute value of DirectWassersteinVariance object taken at β
function (wv::DirectWassersteinVariance)(β::AbstractArray)
    # get residuals, support and indexation
    n = length(wv.Δπ_vect)
    res, supp, idx_ = zeros(wv.E_inds[wv.E+1]), zeros(n), copy(wv.idx)
    get_res_sup_idx!(wv, β, res, supp, idx_)
    
    # compute the wasserstein variance
    wass_var = 0
    for e in 1:wv.E
        temp1 = supp .- res[wv.E_inds[e] .+ idx_[:,e]]
        temp2 = temp1 .* wv.Δπ_vect
        wass_var += wv.w[e] * (temp1' * temp2)
    end 
    
    # return Wasserstein variance
    return wass_var
end

# compute gradient of DirectWassersteinVariance object taken at β
function ProximalOperators.gradient(wv::DirectWassersteinVariance, β::AbstractArray)  
    # get residuals, support and indexation
    n = length(wv.Δπ_vect)
    res, supp, idx_ = zeros(wv.E_inds[wv.E+1]), zeros(n), copy(wv.idx)
    get_res_sup_idx!(wv, β, res, supp, idx_)
    
    # compute the wasserstein variance and gradient
    grad = zeros(length(β))
    for e in 1:wv.E
        temp1 = supp .- res[wv.E_inds[e] .+ idx_[:,e]]
        temp2 = temp1 .* wv.Δπ_vect
        grad .+= (2*wv.w[e]) .* wv.X'[:,wv.E_inds[e] .+ idx_[:,e]] * temp2
    end 
    
    # return gradient 
    return grad
end

# function to compute the residuals, the support of the barycenter, and indexation per env
function get_res_sup_idx!(wv, β, res, supp, idx_)
    # check
    @assert size(wv.X,2) == length(β)
    
    # get residuals
    res .= wv.y .- wv.X*β
    
    # support of the barycenter
    for e in 1:wv.E
        # sorting of the residuals per environment 
        perm = sortperm(res[(wv.E_inds[e] + 1) : wv.E_inds[e+1]])
        idx_[:,e] .= perm[idx_[:,e]]
        supp .+= wv.w[e] .* res[wv.E_inds[e] .+ idx_[:,e]]
    end
end


"""
* ϵₑ: residuals; vector of length nₑ
* kernel: a kernel function
* h: bandwidth 
* residual_diffs: store all differences of sorted residuals
* nₑ: nbr of observations
""" 
# TODO: implement smarter adaptative bandwith
struct Qkde
    residual_diffs
    kernel
    h
    nₑ
    function Qkde(ϵₑ, kernel, h)
        nₑ = length(ϵₑ)
        sorted_residuals = sort(ϵₑ)
        residual_diffs = sorted_residuals[2:end] - sorted_residuals[1:end-1]
        new(residual_diffs, kernel, h * nₑ^(-1/3), nₑ)
    end
end

"""
* x: evaluation probability
"""
function (kde::Qkde)(x)
    nₑ = kde.nₑ
    h = kde.h
    kernel = kde.kernel
    kernel_vect = kernel.((x .- (collect(1:nₑ-1)/nₑ))/h)
    return dot(kde.residual_diffs, kernel_vect) / h
end

# List of kernels
parabolic_func(u) = 0.75 * (1 - u^2) * (-1 <= u <= 1)
gaussian_func(u) = exp(-u^2/2) / sqrt(2*pi) 

"""
* kde: a Qkde type input
* s,t: numbers or vectors of numbers between 0 and 1
* p × p matrix covariance matrix
"""
#cov(kde::Qkde, s::T, t::T) where T <: Real = (min(s,t) - s*t) * kde(s) * kde(t) 
function cov(kde, ss::Vector{T}, ts::Vector{T}) where T <: Real
    p = length(ss)
    smat = repeat(ss',p,1)
    tmat = repeat(ts,1,p)
    # I_s = 0.1 .<= ss .<= 0.9
    # I_t = 0.1 .<= ts .<= 0.9
    # indicator = I_s * I_t'
    return (min.(smat,tmat) - ss*ts') .* (kde.(ss) * kde.(ts)') #.* indicator
end

"""
* kde: a Qkde type input
* nsim: number MC simulations
"""
function calculate_moments(kde, nsim)
    sim = rand(Uniform(), nsim)
    C = cov(kde, sim, sim)
    
    # the mean
    m = tr(C) / nsim
    
    # the variance
    C_nodiag = C - C .* I(nsim)
    cst = 2 / nsim / (nsim-1)
    v = cst * norm(C_nodiag)^2  
    return m, v
end

"""
* ϵ: residuals accross all envrionments
* nₑs: vector of the nbr of obs per env
* α: confidence level
* kernel: a kernle fct for Qkde
"""
function get_asympt_dist(ϵ, nₑs, kernel, h, nsim)
    E = length(nₑs)
    E_inds = vcat([0],cumsum(nₑs))
    n = E_inds[end]
    
    # compute total mean and variance
    kdes = [Qkde(ϵ[(E_inds[e] + 1) : E_inds[e+1]], kernel, h) for e in 1:E]
    q̂(x) =  sum([(nₑs[e] / n) * kdes[e](x) for e in 1:E])
    m, v = calculate_moments(q̂, nsim) .* (E-1)
    
    # gamma threshold
    shape = m^2 / v
    scale = v / m
    
    return Gamma(shape, scale)
end

# bootstrap estimation of the mean and variance of the asymptotic
# distribution of the global minimum WV. 
# Needs to return Gamma(mean, variaance) as output
function bootstrapWV(wv; B = 200, solver = LBFGS(m = 100))
    globWVmin_vect = zeros(B)
    nₑs, E, E_inds = wv.nₑs, wv.E, wv.E_inds
    X, y, n = wv.X, wv.y, E_inds[end]
    for b in 1:B
        # generating bootstrap data
        new_X = copy(X)
        new_y = copy(y)
        for e in 1:E
            rgₑ = (E_inds[e] + 1) : E_inds[e+1]
            idx_selₑ = rand(rgₑ, nₑs[e])
            new_X[rgₑ,:] = X[idx_selₑ,:]
            new_y[rgₑ] = y[idx_selₑ]
        end
        new_wv = DirectWassersteinVariance(new_X, new_y, nₑs, nₑs / n)
        
        # computing global min for wv
        obj = β -> new_wv(β)
        g = β -> gradient(new_wv, β) 
        initial_point = new_X \ new_y
        res = optimize(obj, g, initial_point, solver, inplace=false)
        globWVmin_vect[b] = res.minimum
    end
    m = mean(globWVmin_vect)
    v = var(globWVmin_vect)
    shape = m^2 / v
    scale = v / m
    return Gamma(shape, scale)
end

function optimize_wv(wv, initial_point, solver)
 # Optim.Options(g_tol=1e-12)
    X, y = wv.X, wv.y
    obj = β -> wv(β)
    g = β -> gradient(wv, β) 
    res = optimize(obj, g, initial_point, solver, inplace=false)
    return res
end

function get_lasso(X,y)
    path = fit(Lasso.LassoPath, X, y)
    cf = coef(path, select = MinBIC())
    return cf[2:end]
end

# Random Fourier Featues with Gaussian Kernel
# X: n × p data
# D: number of rand fourier features
# γ: bandwidth of Gaussian kernel, i.e. variance is γ² 
function randfeatures(X, D=30, γ=1)
    n, d = size(X)
    w = randn(d, D)
    b = 2*π*rand(1,D)
    Z = cos.(γ*X*w + ones(n,1) * b)
    return Z
end

function randfeatures(X,w,b,γ)
    n = size(X,1) 
    return cos.(γ*X*w + ones(n,1) * b) 
end

quantiles(x, lb=0.05,ub=0.95,N=18) = quantile(x, LinRange(lb, ub, N + 1)) # requires using Statistics
# f(x_k) = x_k + \sum_i^{N+1} max(0,(x_k - quantile_{lb + ((ub - lb)/N) * i} (x_k)))                    
function simple_splines(X; lb=0.05, ub=0.95, N=18, intercept=true)
    f_xs = []
    for x_k in eachcol(X)
        f_x = [collect(x_k)]
        qs = quantiles(x_k,lb,ub,N)
        for q in qs
            push!(f_x, max.(0, x_k .- q))
        end
        push!(f_xs, f_x...)
    end
    if intercept
        return hcat(ones(size(X,1)), f_xs...)
    end
    return hcat(f_xs...)
    #return hcat([max.(0,x_k .- q) for x_k in eachcol(X) for q in quantiles(x_k,lb,ub,N)]...); so nice so sad
end

# full wasserstein variance, B is the number of bootstrap samples
function WV_screening(wv; α = 0.05 , solver = LBFGS(m = 100), B = 200, nonlinear=false, intercept=true, N=8)
    # input variables
    X, y = wv.X, wv.y
    p = size(X,2)
    h, nsim = 1, 1000 # bandwidth & nbr of MC sims
    
    # init variables
    wvvals_vect, thresholds = zeros(p), zeros(p)
    pvals, causes_vect = zeros(p), zeros(p)
    
    tmp = deepcopy(wv)
    if !nonlinear 
        opt_β, initial_point = zeros(p), X \ y
    else
#         D = D
#         w, b = randn(p, D), 2*π*rand(1,D)
#         R = pairwise(Euclidean(), X, dims=1)
#         γ = sqrt(median(R) / 2) # median heuristic for bandwidth; see https://arxiv.org/pdf/1707.07269.pdf
        tmp.X = simple_splines(X, intercept=intercept, N=N)
        initial_point = tmp.X \ y
        opt_β = zeros(length(initial_point))
    end
    global_min = 0
    dists = Vector{UnivariateDistribution}(undef,p)

    # loop
    unsolved = true
    while unsolved 
        println("______________________________________")
        println("-          Initialization...          ")
        
        # first computation of global optimum
        res = optimize_wv(tmp, initial_point, solver)
        opt_β = res.minimizer
        global_min = res.minimum
               
        for i in 1:p 
        
            println("______________________________________")
            println("-         Treating variable $i        ")
            
            # create objects
            select = filter(x->x!=i, 1:p)
            wv_sub = deepcopy(wv)
            if !nonlinear 
                wv_sub.X = X[:,select]
                res = optimize_wv(wv_sub, opt_β[select], solver)
            else
                if intercept
                    inds = (i-1)*N+2:i*N+1
                end
                inds = (i-1)*N+1:i*N+1
                wv_sub.X = tmp.X[:, Not(inds)] # N is the number of basis functions 
                res = optimize_wv(wv_sub, opt_β[Not(inds)], solver)
            end
            
            # get minimizer
            min_i = res.minimum
#             if min_i < global_min 
#                println("Restarting optimization")
#                tmp = zeros(p)
#                tmp[select] = res.minimizer
#                initial_point = tmp
#                break
#             end
#             if i == p; unsolved = false; end

#             collect the 1 - α quantiles
#             dist =  get_asympt_dist(y - wv_sub.X * res.minimizer,
#                   wv.nₑs, parabolic_func, h, nsim)
#             dists[i] = dist
            wvvals_vect[i] = min_i
        end
        unsolved = false
    end
    
    # bootstrap correction
    factor = 1
    if B > 0
        #dist =  get_asympt_dist(y - X * opt_β, 
        #            wv.nₑs, parabolic_func, h, nsim) # may change later with calculate_moments...
        #factor = bootstrapWV(wv; B = B, solver = solver) / (dist.α * dist.θ)
        distr = bootstrapWV(tmp; B = B, solver = solver) #/ (dist.α * dist.θ)
    end
    thresholds = quantile(distr, 1 .- α)
    pvals = 1 .- cdf.(distr, wvvals_vect)
    causes_vect = [(wvvals_vect .> t) for t in thresholds]
    # if only one \alpha is provided treat causes_vect as single array.
    if length(causes_vect) == 1; causes_vect = causes_vect[1] end 
    
    
    
    #thresholds = quantile.(dists, 1 - α) .* factor  
    #pvals = 1 .- cdf.(dists,wvvals_vect ./ factor)
    #causes_vect = (wvvals_vect .> thresholds)
    
    return wvvals_vect, causes_vect, global_min, pvals, thresholds
end
