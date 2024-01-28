#  testing statistical significance of BQN calibrated forecasts using
#    - Diebold-Mariano hypothesis test of equal CRPS for each lead site and lead time
#    - Benjamini and Hochberg (1995) approach to aggregate
#
#   julia --project=./   # 64 GB memory is probably required
#   

using DataFrames, CSV, Statistics, Distributions, NamedArrays


include("loss_functions.jl")


const models = ["pangu", "hres", "ens", "meps", "ens0", "meps0"]
const prm    = ["t2", "ws10"]


#  function to read all forecasts for a given model (9 experiments)
function get_forecasts(prm, model)
    experiments = [string("m",i,"b",j) for j in 1:3, i in 1:3][:]
    files = string.("data/qts_bqn_$(prm)_+60_$(model)_", experiments, ".csv")
    df = CSV.read.(files, DataFrame)
    return df 
end


#  Benjamini and Hochberg (1995) procedure for false discovery rate at given level 
function bh_test(p; level = 0.05)
    M = length(p)
    psort = sort(p)
    padj = findmax(psort[psort .< (level/M .* (1:M))])[1]
    return p .<= padj
end
    

#  function to compute statistics by site and lead time
function compute_stats(df, df2; level = 0.05)
    ens = string.("Q", 0:100)
    crps = Array{Float32,3}(undef, nrow(df[1]), length(df), 2)
    Threads.@threads for i in eachindex(df)
        crps[:, i, 1] = crps_ensemble(Matrix(df[i][:, ens]), df[i].obs; agg = identity)
        crps[:, i, 2] = crps_ensemble(Matrix(df2[i][:, ens]), df2[i].obs; agg = identity)
    end
    crps = mean(crps, dims = 2)
    scr = DataFrame(site = df[1].site, lt = df[1].lt,
                    crps1 = crps[:,1,1], crps2 = crps[:,1,2])
    scr.diff  = scr.crps1 .- scr.crps2
    scr.diff2 = scr.diff.^2
    
    scr_grp = groupby(scr, [:site, :lt])
    scr = combine(scr_grp, [:crps1, :crps2, :diff, :diff2] .=> mean, :crps1 .=> length => :n)
    scr.dm = @. sqrt(scr.n) * (scr.crps1_mean - scr.crps2_mean) / scr.diff2_mean
    scr.p_value = cdf.(Normal(0,1), scr.dm)
    scr.bh_5pct = bh_test(scr.p_value)
    
    return scr
end


function significance_stats()
    scores = Array{DataFrame, 3}(undef, length(models), length(models), 2)
    for (ip, p) in enumerate(prm)
        println("\nparameter: $(p)")
        for (i, m) in enumerate(models)
            df = get_forecasts(p, m)
            for (i2, m2) in enumerate(models)
                if m != m2
                    df2 = get_forecasts(p, m2)
                    scores[i, i2, ip] = compute_stats(df, df2; level = 0.05)
                    println("  $(m) vs $(m2): $(round(mean(scores[i,i2,ip].bh_5pct), digits=2))")
                    df2 = nothing
                    GC.gc()
                end
            end
        end
    end
    return scores
end


@time stats = significance_stats()  # ~3 hours
JLD2.@save "data/significance_cal.jld2" models prm stats

best = NamedArray(zeros(6,6,2), (models, models, prm))
for p in axes(stats, 3)
    for m in axes(stats, 1)
        for m2 in axes(stats, 2)
            if m != m2
                best[m,m2,p] = round(mean(stats[m,m2,p].bh_5pct)*100, digits=1)
            end
        end
    end
end

         



#=
:, :, C=t2] =
A ╲ B │ pangu   hres    ens   meps   ens0  meps0
──────┼─────────────────────────────────────────
pangu │   0.0   33.6   28.4   14.4   42.5   20.3
hres  │  27.9    0.0   23.7    9.3   44.3   17.6
ens   │  39.1   46.1    0.0   16.1   91.5   26.1
meps  │  45.5   49.2   40.1    0.0   53.6   72.2
ens0  │  20.9   23.1    1.6    6.3    0.0   11.9
meps0 │  28.9   32.2   24.8    3.6   36.3    0.0

[:, :, C=ws10] =
A ╲ B │ pangu   hres    ens   meps   ens0  meps0
──────┼─────────────────────────────────────────
pangu │   0.0   27.5   15.2    7.0   32.8   20.6
hres  │  28.9    0.0   18.1    9.2   36.6   24.9
ens   │  50.4   54.5    0.0   19.3   91.6   40.5
meps  │  54.8   52.1   42.3    0.0   56.4   88.2
ens0  │  27.7   31.0    2.8    8.3    0.0   23.6
meps0 │  23.9   24.4   15.9    0.5   27.5    0.0

=#



