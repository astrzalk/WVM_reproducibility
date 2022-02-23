using Serialization, DataFrames, StatsPlots, Statistics
using LaTeXStrings


# Convert an array (length n) of k-tuples `xs` to a matrix of size k × n i.e. (nfps, nfns, len(S), len(Ŝ)) by nsims
# Taken from https://stackoverflow.com/questions/67137544/convert-a-vector-of-tuples-in-an-array-in-julia
vectuples_tomat(xs) = hcat(collect.(xs)...)

# for some of the errors I had to skip the first numvar (2 variables, because lasso someimtes only selected one var in this case causing wvm to crash.)
function fix_list(xss, num=8)
    for xs in xss
        if length(xs) == num
            if typeof(xs[1]) <: Tuple
                pushfirst!(xs, (0,0,0))
            elseif typeof(xs[1]) <: Float64
                pushfirst!(xs, 0.0)
            end
        end
    end
    hcat(xss...) # returns #number of pre-selected vars tried × nsims
end

function extract_mat(X,which_entry=1)
    M = zeros(Int8, size(X))
    for j in 1:size(X,2)
        for i in 1:size(X,1)
            M[i,j] = X[i,j][which_entry]
        end
    end
    return collect(M')
end

# Assume order of files is: ols, lingam, gies, icp, wvm
function extract_dataframe(datapath, files, methodnames=["OLS", "LiNGAM", "GIES", "ICP", "WVM"])
    n = length(deserialize(datapath * files[1])) # extract nsims
    paths = datapath .* files
    results = hcat([vectuples_tomat(deserialize(p)) for p in paths]...) # (nfps, nfns, len(Ŝ)) × length(methodnames)*nsims matrix
    methodlabels = repeat(methodnames, inner=n)
    colnames = ["FalsePositives", "FalseNegatives", "NumberofInferredCauses"]
    df = insertcols!(DataFrame(collect(results'), colnames),  size(results,1)+1, :Method => methodlabels)
    return df
end

function diff_alpha_results(fp,colnames = ["FalsePositives", "FalseNegatives", "NumberofInferredCauses"], alphas=[string(i) for i in 0.1:0.1:0.9])
    D = vectuples_tomat(deserialize(fp))
    n = size(D,2)
    fps, fns, numinfs = [], [], []
    for row in eachrow(D) # assume it is alphas by samples (≈ 9 × 100)
        fp, fn, numinf = [], [], []
        for r in row
            push!(fp, r[1])
            push!(fn, r[2])
            push!(numinf, r[3])
        end
        push!(fps, fp)
        push!(fns, fn)
        push!(numinfs, numinf)
    end
    alphalabels = repeat(alphas, inner=n)
    df = insertcols!(DataFrame(hcat(vcat(fps...), vcat(fns...), vcat(numinfs...)), colnames), length(colnames) + 1, :Alpha => alphalabels)
    return df
end

function add_stat_columns(df, p, numS=6, normalize=true)
    error = (df[!,:FalsePositives] + df[!,:FalseNegatives]) / p
    if normalize
        fpr = df[!,:FalsePositives] / (p - numS)
        fnr = df[!,:FalseNegatives] / (p - numS)
    else
        fpr = df[!,:FalsePositives]
        fnr = df[!,:FalseNegatives]
    end
    insertcols!(df, size(df,2)+1, :Error => error)
    insertcols!(df, size(df,2)+1, :FPR => fpr)
    insertcols!(df, size(df,2)+1, :FNR => fnr)
    return df
end

# datapath = "../src/results/numS-6/"
# fn = "normal-diff-alphas-wvm-results.jls"
# test = deserialize(datapath * fn)
# FP, FN = extract_alpha_stats(test)
function extract_alpha_stats(xss, numalpha=9)
    FP = zeros(length(xss), numalpha)
    FN = zeros(length(xss), numalpha)
    for (i, xs) in enumerate(xss)
       for (j, x) in enumerate(xs)
            FP[i,j] = x[1]
            FN[i,j] = x[2]
        end
    end
    return FP, FN
end


# global_unique_pvals: sorted unique pvalues from all 100 simulations
# sim_iscause_pvals: sorted by pvalue 2 column matrix; first column iscause and second column pvalue from one simulation
# output: number of global unique pvals by prec, recall
function prcurve_per_sim(global_unique_pvals, sim_iscause_pvals, numS)
    n,p = length(global_unique_pvals), size(sim_iscause_pvals,1)
    M = zeros(n, 2) # tp fp mat for all pvals
    i,j = 1,1
    while j <= p
        if global_unique_pvals[i] < sim_iscause_pvals[j,2]
            i += 1
            M[i,:] .= M[i-1,:]
        else 
            Bool(sim_iscause_pvals[j,1]) ? M[i,1] += 1 : M[i,2] += 1 
            j += 1
        end
    end
    recall = M[:,1] / numS
    precision = M[:,1] ./ (M[:,1] .+ M[:,2])
    precision[isnan.(precision)] .= 1
    return hcat(precision, recall)
end

function prcurve(df, numS=6, p=50, numsim=100)
    n = size(df,1)
    @assert n == p * numsim
    df[:,:pvals] .+= eps(Float64)# adding machine epsilon because wvm's pvalues are too small
    global_pvals = sort(df[:,:pvals])

    pushfirst!(global_pvals, 0)

    pr_mats = []
    for sim_i_index in Iterators.partition(collect(1:n), p)
        sim_i_iscause_pvals = Matrix(sort(df[sim_i_index, [:iscause, :pvals]], :pvals))
        pr_mat = prcurve_per_sim(unique(global_pvals), sim_i_iscause_pvals, numS)
        push!(pr_mats, pr_mat)
    end

    return sum(pr_mats) / numsim
end


function our_boxplot(df, errortype, label, ylab, xlab="", legend=true, group=:Method, leg_position=:topright, dpi=100, addtilt=0)
    # In order to order these not alphabetically I needed to hack the boxplots.jl source in StatsPlots (long term I will make a recipe to do this).
    xs = collect(1:length(unique(df[!,group]))) .- 0.5
    gdf = groupby(df, group)
    metric_means = combine(gdf, errortype => mean)[!, 2]
    if legend
        @df df boxplot(cols(group), cols(errortype), linewidth=4, color=:lightgrey, xlabel=xlab, ylabel=ylab, label="",
        xtickfontsize=18, ytickfontsize=18, guidefontsize=18, legendfontsize=15, legend=leg_position, dpi=dpi, xrot=addtilt)
        scatter!(xs, metric_means, color=:yellow, markersize=8, msw=3, markershape=:diamond, label=label)
    else
        @df df boxplot(cols(group), cols(errortype), linewidth=4, color=:lightgrey, xlabel=xlab, ylabel=ylab, label="",
        xtickfontsize=18, ytickfontsize=18, guidefontsize=18, legendfontsize=15, legend=legend, dpi=dpi, xrot=addtilt)
        scatter!(xs, metric_means, color=:yellow, markersize=8, msw=3, markershape=:diamond, label=label)
    end
end

function plot_prcurve(pr_1, pr_2, labs=["ICP", "WVM"])
    plot(pr_1[:,2], pr_1[:,1], xlabel="Recall", ylabel="Precision", linetype=:steppre, 
linewidth=6,xtickfont=font(18),ytickfont=font(18), guidefont=font(18), label=labs[1])
    plot!(pr_2[:,2], pr_2[:,1], xlabel="Recall", ylabel="Precision", linetype=:steppre, linewidth=6,
xtickfont=font(18),ytickfont=font(18), guidefont=font(18), label=labs[2], legend=:bottomleft,legendfontsize=14, xlim=[0,1.005], dpi=100)  
end


