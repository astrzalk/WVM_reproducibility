include("SimulationsUtils.jl")
include("MethodUtils.jl")
include("WassersteinVariance.jl")
using Serialization 

function benchmark(nsim, numS, ne=500, noisetype="normal", numvar=18, seed=123)
    Random.seed!(seed)

    ######################## SIMULATION ##############################
    # nsim = 100
    # These will store a four tuple for each simulation: nfp, nfn, length(S), length(Ŝ) 
    ols_errs = []
    lingam_errs = []
    gies_errs = []
    icp_errs = []
    wvm_errs = []

    simtype = "numS-$(numS)/ne-$(ne)/$(noisetype)"
    for i in 1:nsim
        ############### DATA SIM #################
        # collect simulation data
        X_full, S, nₑs = deserialize("../data/Simulations/$(simtype)/sim-$(i).jls") #"Simulations/$(noisetype)/numS-$(numS)/sim-$(i)$(addedinfo).jls"
        n, p = size(X_full,1), size(X_full,2)
        X, y = X_full[:, 1:end-1], X_full[:, end]
        E = length(nₑs)
        
        ################# OLS #################
        mm = lm(hcat(X, ones(size(X,1))),y)
        Ŝ_ols = sig_pvals_ols(mm, 0.1/(p-1))
        push!(ols_errs, error_count(S, Ŝ_ols))
        
        ################# LinGam #################
        Ŝ_lingam = lingam(X_full)
        push!(lingam_errs, error_count(S, Ŝ_lingam))
        
        ################# GIES #################
        Ŝ_gies = gies(X_full, E, nₑs[1])
        push!(gies_errs, error_count(S, Ŝ_gies))
        
        ################# ICP #################
        α = 0.1
        icp_res = icp(X, y, nₑs, α, true)
        Ŝ_icp = findall_nonzero(isnothing(icp_res) ? zeros(p-1) : icp_res)
        push!(icp_errs, error_count(S, Ŝ_icp))
        
        ################# lasso #################
        cf = select_lasso_by_num_var(X, y, numvar)
        select = 1:p-1
        select = filter(x -> cf[x] != 0, select)
        
        ################# WVM #################
        wv = DirectWassersteinVariance(X[:,select], y, nₑs, nₑs / n)
        res = WV_screening(wv, α=α, solver=LBFGS(m=100), B=50)
        Ŝ_wvm = select[findall_nonzero(res[2])]
        push!(wvm_errs, error_count(S, Ŝ_wvm))
        
        println("###### SIMULATION $i IS DONE ! ######")
    end 
    run(`mkdir -p ../data/results/$(simtype)`)
    serialize("../data/results/$(simtype)/ols.jls", ols_errs)
    serialize("../data/results/$(simtype)/lingam.jls", lingam_errs)
    serialize("../data/results/$(simtype)/gies.jls", gies_errs)
    serialize("../data/results/$(simtype)/icp.jls", icp_errs)
    serialize("../data/results/$(simtype)/wvm.jls", wvm_errs)
end
benchmark(100, 6, 500, "normal")
benchmark(100, 6, 500, "mixed")
