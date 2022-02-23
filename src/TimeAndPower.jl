include("SimulationsUtils.jl")
include("MethodUtils.jl")
include("WassersteinVariance.jl")
using Serialization 

function time_power_wvm_icp(nsim, numS, ne=500, noisetype="normal", seed=123)
    Random.seed!(seed)

    ######################## SIMULATION ##############################
    # These will store a four tuple for each simulation: nfp, nfn, length(S), length(Ŝ) 
    total_icp_errs, total_time_icp = [], []
    # total_wvm_errs, total_time_wvm = [], []
    
    simtype = "numS-$(numS)/ne-$(ne)/$(noisetype)"
    for i in 1:nsim
        println("--------- Sim $(i) ----------")
        X_full, S, nₑs = deserialize("../data/Simulations/$(simtype)/sim-$(i).jls")
        p = size(X_full,2)
        X = X_full[:, 1:end-1]
        y = X_full[:, end]
        n = size(X, 1)
        
        icp_errs, time_icp = [], []
        # wvm_errs, time_wvm = [], []
        for numvar in 2:2:30
            print("------ Number of Selected Vars: $(numvar) -------")
            ################# lasso #################
            cf = select_lasso_by_num_var(X, y, numvar)
            select = filter(x -> cf[x] != 0, 1:p-1)
            if length(select) <= 1; continue end
            X_select = X[:,select]

            ################# ICP #################
            α = 0.1
            if numvar <= 18
                t_icp = @elapsed begin 
                    Ŝ_icp = select[icp(X_select, y, nₑs, α, false, "all")]
                end
                push!(icp_errs, error_count(S, Ŝ_icp))
                push!(time_icp, t_icp)
            end
        
            ################# WVM #################
            t_wvm = @elapsed begin
                wv = DirectWassersteinVariance(X_select, y, nₑs, nₑs / n)
                res = WV_screening(wv, α=α, solver=LBFGS(m=100), B=50)
                Ŝ_wvm = select[findall_nonzero(res[2])]
            end
            push!(wvm_errs, error_count(S, Ŝ_wvm))
            push!(time_wvm, t_wvm)
        
            println("###### SIMULATION $i IS DONE ! ######")
        end
        push!(total_icp_errs, icp_errs)
        push!(total_time_icp, time_icp)
        
        push!(total_wvm_errs, wvm_errs)
        push!(total_time_wvm, time_wvm) 
    end
    serialize("../data/results/$(simtype)/icp-time-power.jls", (total_icp_errs, total_time_icp))
    serialize("../data/results/$(simtype)/wvm-time-power.jls", (total_wvm_errs, total_time_wvm))
end
time_power_wvm_icp(100, 6)
