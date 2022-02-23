include("SimulationsUtils.jl")
include("MethodUtils.jl")
import ARCHModels
using Serialization 

function simulation(nsim, numS, ne=500, noisetype="normal", seed=123)
    Random.seed!(seed)

    ######################## SIMULATION ##############################
    # SCM parameters 
    p_anterior, p_posterior = 20, 30
    k, num_parents_target = 12, numS
    lb_coef, ub_coef = 0.2, 1
    var_min, var_max = 0.3^2, 1^2

    p = p_anterior + p_posterior + 1
    target = p_anterior + 1
    E = 5

    for i in 1:nsim    
        # generate the linear SCM and its interventional environments
        test_scm = high_dim_SCM(p_anterior, p_posterior, k, num_parents_target, 
            lb=lb_coef, ub=ub_coef, var_min=var_min, var_max=var_max)

        inter_scm = generate_interLinAddSCM(true,E,1.0; scm=test_scm, additive=false, overlap = 0.4,
            inter_noise_selection_params = Dict("prob_cst" => 1/3, "lb" => 0.5, "ub" => 5, "flip" =>false), target=target)
        
        # simulate the data
        nₑs = repeat([ne], E)
        if noisetype == "mixed"
            ds = [Normal(), ARCHModels.StdT(3), ARCHModels.StdT(5), ARCHModels.StdT(10), 
                  ARCHModels.StdT(20), ARCHModels.StdT(50), Uniform(-sqrt(3), sqrt(3))] 
        elseif noisetype == "normal"
            ds = [Normal()]
        else
            ds = [Normal()]
        end
        X, y = simulate(inter_scm, nₑs, ds)
        X_full = hcat(X,y)
        
        ################# COLLECT BETA #################
        β_true = causalcoeff(test_scm, target)
        S = findall_nonzero(β_true) 
        
        # save simulation for later (when we have more time for ICP runs)
        run(`mkdir -p ../data/Simulations/numS-$(numS)/ne-$(ne)/$(noisetype)`)
        serialize("../data/Simulations/numS-$(numS)/ne-$(ne)/$(noisetype)/sim-$(i).jls", (X_full, S, nₑs))
    end
end
println("Generating Simulation normal distributions, ne = 100, numS = 6")
simulation(100, 6, 100, "normal")
println("Generating Simulation normal distribution, ne = 500, numS = 6")
simulation(100, 6, 500, "normal")
println("Generating Simulation mixed distributions, ne = 500, numS = 6")
simulation(100, 6, 500, "mixed")
println("Generating Simulation normal distribution, ne = 500, numS = 12")
simulation(100, 12, 500, "normal")
