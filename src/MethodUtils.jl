using RCall 
# To install pcalg on your R installation you need to do the following:
# In R (tested on version 4.0.2) do the following commands:
# if (!requireNamespace("BiocManager", quietly = TRUE))
#       install.packages("BiocManager")
# BiocManager::install(version = "3.13")
# BiocManager::install(c("graph", "RBGL"))
# install.packages(pcalg)
# install.packages(InvariantCausalPrediction)
using StatsBase
using Lasso, GLM
using SparseArrays # for lasso funciton (nnz)
using Random

findall_nonzero(xs) = findall(x->!iszero(x), xs)

# error count function
function error_count(S, Ŝ)
    nfn = length(S) - length(Ŝ ∩ S) # false neg
    nfp = length(Ŝ) - length(Ŝ ∩ S) # false pos
    return nfp, nfn, length(Ŝ)
end

# Tool to extract pvals from ols model hacked form GLM source code
# Only neeeds Lasso even though we use GLM.coeftable
function sig_pvals_ols(mm, α)
    ps = GLM.coeftable(mm).cols[4]
    return findall(x -> x < α, ps)
end

# A method to extract a lasso path by choosing an upperbound
# on the number of varriables a path finds. Concretely, it returns 
# coefficients for the last path that has at most numvar variables.
function select_lasso_by_num_var(X, y, numvar)
    path = fit(Lasso.LassoPath, X, y)
    coefs = path.coefs

    # Fast way to compute number of coefficients w.r.t to sparse matrix representation, grabed from lasso sparse
    ncoefs = zeros(Int, size(coefs, 2))
    for i = 1:size(coefs, 2)-1
        ncoefs[i] = coefs.colptr[i+1] - coefs.colptr[i]
    end
    ncoefs[end] = nnz(coefs) - coefs.colptr[size(coefs, 2)] + 1

    ind = findall(x -> x <= numvar, ncoefs)
    # cfs = coefs[:,2:end]  # no intercept
    # return cfs[:,ind[end]]
    return coefs[:,ind[end]]
end

# Assume X has the target in the last column.
function lingam(X, return_coefs=false)
    R"""
    library("pcalg")
    set.seed(1234)
    # LinGAM
    p <- dim($X)[2]
    res <- lingam($X)
    B <- res["Bpruned"]
    lingam_res <- B[[1]][p,]
    """
    @rget lingam_res;
    if return_coefs; return lingam_res; end
    return findall_nonzero(lingam_res)
end


# Assume X has the target in the last column.
# -> nobs refers to the number of samples in the observational
#    dataset, that is from the first environment.
function gies(X, E, nobs)
    R"""
    library("pcalg")
    # GIES
    n <- dim($X)[1]
    p <- dim($X)[2]
    inter_targets <- vector("integer", length = p - 1)
    for (i in 1:p - 1) {inter_targets[i] <- i}
    # Always assume two environments even if we have more,
    # since we are intervening on all variables in the other interventional environments.
    targetInds <- rep(1:2, c($nobs, n-$nobs))
    score <- new("GaussL0penIntScore", $X, list(c(integer(0)), inter_targets), targetInds)
    gies_fit <- gies(score)
    in_edges <- gies_fit[["essgraph"]][[".->.in.edges"]]
    """
    @rget in_edges
    @rget p
    tar_out_edges = Set([i for (i,v) in enumerate(values(in_edges)) if p in v])
    target_parents = setdiff(Set(in_edges[Symbol(string(p))]), tar_out_edges) # remove all outgoing edges, i.e. unoriented edges
    return sort(collect(target_parents))
end

function icp(X, y, nes, α, return_coefs=false, selection="boosting", return_pvals=false)
    p = size(X,2)
    R"""
    library("InvariantCausalPrediction")
    # ICP
    x <- $X
    y <- $y
    E <- length($nes)
    ExpInd <- rep(1:E, $nes)
    icp <- ICP(x,y, ExpInd, alpha=$(α), test="ks", showAcceptedSets=FALSE, showCompletion=FALSE, selection=$selection)
    icp_res <- icp$maximinCoefficients
    ps <- icp$pvalues
    """
    @rget icp_res
    @rget ps;
    if return_coefs; return icp_res; end
    if return_pvals; return findall_nonzero(isnothing(icp_res) ? zeros(p) : icp_res), ps; end
    return findall_nonzero(isnothing(icp_res) ? zeros(p) : icp_res)
end

# extract error ratios and fpr from serialized output (sim_output)
# p is the number of predictors
# numS is the number of direct causes
function extract_errors(sim_output, p, numS)
    err_ratio_list, fpr_list = [], []
    for errors in sim_output
        push!(err_ratio_list, (errors[1] + errors[2])/p)
        push!(fpr_list, errors[1]/(p - numS))
    end
    return err_ratio_list, fpr_list
end
