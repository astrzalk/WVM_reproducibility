include("SimulationsUtils.jl")
include("MethodUtils.jl")
include("WassersteinVariance.jl")
using Serialization 

function compare_wvm_icp(nsim, numS, ne, noisetype="normal", numvar=18, seed=123)
    Random.seed!(seed)
    ######################## SIMULATION ##############################
    # nsim = 100
    # These will store a four tuple for each simulation: nfp, nfn, length(Ŝ) 
    errs = []
    iscauses, wvm_pvals, icp_pvals = [], [], []
    simtype = "numS-$(numS)/ne-$(ne)/$(noisetype)"
    for i in 1:nsim
        ############### DATA SIM #################
        # collect simulation data
        X_full, S, nₑs = deserialize("../data/Simulations/$(simtype)/sim-$(i).jls")
        n, p = size(X_full,1), size(X_full,2)
        X, y = X_full[:, 1:end-1], X_full[:, end]

        iscause = zeros(p-1)
        iscause[S] .= 1
        
        ################# lasso #################
        cf = select_lasso_by_num_var(X, y, numvar)
        select = 1:p-1
        select = filter(x -> cf[x] != 0, select)
        
        ################# WVM + ICP #################
        αs = [0.1, 0.3, 0.5, 0.7]
        wv = DirectWassersteinVariance(X[:,select], y, nₑs, nₑs / n)
        res = WV_screening(wv, α=αs, solver=LBFGS(m=100), B=50)
        Ŝs_wvm = [select[findall_nonzero(r)] for r in res[2]]
        wvm_errs = [error_count(S, Ŝ_wvm) for Ŝ_wvm in Ŝs_wvm]
        wvm_pval = ones(p-1)
        wvm_pval[select] .= res[4]

        icp_res, icp_pval = icp(X, y, nₑs, αs[1], false, "boosting", true)
        Ŝ_icp = icp_res # findall_nonzero(isnothing(icp_res) ? zeros(p-1) : icp_res)
        err = push!([error_count(S, Ŝ_icp)], wvm_errs...)
        push!(errs, err)

        push!(icp_pvals, icp_pval)
        push!(wvm_pvals, wvm_pval)
        push!(iscauses, iscause)
        println("###### SIMULATION $i IS DONE ! ######")
    end 
    pr_curve_res_wvm = hcat(vcat(iscauses...), vcat(wvm_pvals...))
    pr_curve_res_icp = hcat(vcat(iscauses...), vcat(icp_pvals...))
    run(`mkdir -p ../data/results/$(simtype)`)
    # serialize("results/numS-$(numS)/$(noisetype)$(addedinfo)-wvm-and-icp-results.jls", errs)
    serialize("../data/results/$(simtype)/diff-alpha-wvm-and-icp.jls", errs)
    serialize("../data/results/$(simtype)/pr-curve-wvm.jls", pr_curve_res_wvm)
    serialize("../data/results/$(simtype)/pr-curve-icp.jls", pr_curve_res_icp)
end
compare_wvm_icp(100, 6, 100, "normal")
compare_wvm_icp(100, 6, 500, "normal")
compare_wvm_icp(100, 12, 500, "normal")
