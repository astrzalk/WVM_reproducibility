include("SimulationsUtils.jl")
include("MethodUtils.jl")
include("WassersteinVariance.jl")
using Serialization 

function diff_alpha_results(nsim, which_method=:WVM, numS=6, ne=500, noisetype="normal", numvar=18, seed=123)
    Random.seed!(seed)
    ######################## SIMULATION ##############################
    # nsim = 100
    # These will store a four tuple for each simulation: nfp, nfn, length(Ŝ) 
    wvm_errs = []
    icp_errs = []
    simtype = "numS-$(numS)/ne-$(ne)/$(noisetype)"
    for i in 1:nsim
        ############### DATA SIM #################
        # collect simulation data
        X_full, S, nₑs = deserialize(".../data/Simulations/$(simtype)/sim-$(i).jls")
        n, p = size(X_full,1), size(X_full,2)
        X, y = X_full[:, 1:end-1], X_full[:, end]
        
        ################# lasso #################
        cf = select_lasso_by_num_var(X, y, numvar)
        select = 1:p-1
        select = filter(x -> cf[x] != 0, select)
        
        ################# WVM #################
        if which_method == :WVM
            αs = collect(0.1:0.1:0.9)
            wv = DirectWassersteinVariance(X[:,select], y, nₑs, nₑs / n)
            res = WV_screening(wv, α=αs, solver=LBFGS(m=100), B=50)
            Ŝs_wvm = [select[findall_nonzero(r)] for r in res[2]]
            push!(wvm_errs, [error_count(S, Ŝ_wvm) for Ŝ_wvm in Ŝs_wvm])
        elseif which_method == :ICP
            errs = []
            for α in 0.1:0.1:0.9
                icp_res = icp(X, y, nₑs, α, true)
                Ŝ_icp = findall_nonzero(isnothing(icp_res) ? zeros(p-1) : icp_res)
                push!(errs, error_count(S, Ŝ_icp))
            end
            push!(icp_errs, errs)
        end
        println("###### SIMULATION $i IS DONE ! ######")
    end 
    run(`mkdir -p ../data/results/$(simtype)`)
    if which_method == :WVM
        serialize("../data/results/$(simtype)/diff-alphas-wvm.jls", wvm_errs)
    elseif which_method == :ICP
        serialize("../data/results/$(simtype)/diff-alphas-icp.jls", icp_errs)
    end
end
diff_alpha_results(100)
