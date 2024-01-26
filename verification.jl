#  forecast verification
# 
#  files at /lustre/storeB/users/johnbb/prosjekt/pangu-pp/qts_bqn_*
#  julia --project=./
#


using CSV, DataFrames, Statistics, Dates
using JLD2


include("loss_functions.jl")


models = ["pangu", "hres", "meps0", "ens0", "meps", "ens"]
prm    = ["t2", "ws10"]
max_lt = 60



#
#    B Q N     F O R E C A S T S
#

prob = Float32.(0:0.01:1)
kqts = string.("Q", 0:100)    
lts  = collect(6:6:60)
qs   = zeros(Float32, length(lts), 3, 3, length(models), 2);
crps = zeros(Float32, length(lts), 3, 3, length(models), 2); 
mae  = zeros(Float32, length(lts), 3, 3, length(models), 2);
sde  = zeros(Float32, length(lts), 3, 3, length(models), 2);
sd   = zeros(Float32, length(lts), 3, 3, length(models), 2);
sharp = zeros(Float32, length(lts), 3, 3, length(models), 2);
sharp_sd = zeros(Float32, length(lts), 3, 3, length(models), 2);
for (ip, p) in enumerate(prm)
    println(p)

    for (j, mod) in enumerate(models)
        println("  ", mod)

        for b in 1:3     # bootstrap

            for m in 1:3   # bqn model
                
                d = CSV.read("./data/qts_bqn_$(p)_+$(max_lt)_$(mod)_m$(m)b$(b).csv", DataFrame)
                for (il, l) in enumerate(lts)
                    kc  = d.lt .== l
                    ens = Matrix(d[kc, kqts])
                    qs[il, b, m, j, ip]   = qtloss(ens', d.obs[kc], prob)
                    crps[il, b, m, j, ip] = crps_ensemble(ens, d.obs[kc])
                    mae[il, b, m, j, ip]  = mean(abs.(d.Q50[kc] .- d.obs[kc]))
                    sde[il, b, m, j, ip]  = std(abs.(d.Q50[kc] .- d.obs[kc]))
                    grp = groupby(d[kc,:], :site)
                    sd[il, b, m, j, ip]  = mean(combine(grp, :Q50 => std).Q50_std ./ combine(grp, :obs => std).obs_std)
                    sharp[il, b, m, j, ip]  = mean(std(ens, dims=1))
                    sharp_sd[il, b, m, j, ip]  = std(std(ens, dims=1))
                end

            end
            
        end
        
    end

end

scores = (qs = qs, crps = crps, mae = mae, sde = sde, sd = sd, sharpness = sharp, sharpness_sd = sharp_sd,
          lts = lts, models = models, prm = prm,
          dims = ["lts","BQN model","bootstrap","NWP model","parameter"])
dropdims(mean(scores.crps, dims=2:3); dims=(2,3))
dropdims(mean(scores.qs, dims=2:3); dims=(2,3))
dropdims(mean(scores.mae, dims=2:3); dims=(2,3))
dropdims(mean(scores.sde, dims=2:3); dims=(2,3))
dropdims(mean(scores.sharpness, dims=2:3); dims=(2,3))
dropdims(mean(scores.sharpness_sd, dims=2:3); dims=(2,3))
JLD2.@save "./data/scores_bqn_+$(max_lt).jld2" scores




#
#    R A W    F O R E C A S T S
#

nwp = JLD2.load("./data/nwp+obs.jld2");
lts = unique(nwp["data"][1].lt)
models = ["pangu", "hres", "meps0", "ens0", "meps", "ens"]
mae    = zeros(Float32, length(lts), length(models), length(prm));
bias   = zeros(Float32, length(lts), length(models), length(prm));
sde    = zeros(Float32, length(lts), length(models), length(prm));
dmax   = zeros(Float32, length(lts), length(models), length(prm));
dmin   = zeros(Float32, length(lts), length(models), length(prm));
rmax   = zeros(Float32, length(lts), length(models), length(prm));
sd     = zeros(Float32, length(lts), length(models), length(prm));
crps   = zeros(Float32, length(lts), length(models), length(prm));    
sharp  = zeros(Float32, length(lts), length(models), 2);
sharp_sd = zeros(Float32, length(lts), length(models), 2);
for (j, m) in enumerate(models)

    # read data
    print(m, ": ")
    d = nwp["data"][indexin([m], nwp["models"])][1]
    println(size(d))
    filter!(row -> year(DateTime(row.time_ref)) == 2022, d)
    k = map(u -> eltype(u) == Float64, eachcol(d))
    d[!, k] = Float32.(d[:, k])
      
    # compute medians for deterministic forecast
    if m in ["meps", "ens"]
        mbr = m == "ens" ? (0:50) : (0:14)
        k   = Symbol.("ws10_", mbr)
        d.ws10 = median(Matrix(d[:, k]), dims = 2)[:]
        k = Symbol.("t2_", mbr)
        d.t2 = median(Matrix(d[:, k]), dims = 2)[:]
    end

    # compute error
    d.t2_error   = d.t2 .- 273.15f0 .- d.obs_t2
    d.ws10_error = d.ws10 .- d.obs_ws10

    for (ip, p) in enumerate(prm)
        if p == "ws10"
            for (i, l) in enumerate(lts)
                kc = d.lt .== l
                mae[i, j, ip]  = mean(abs.(d.ws10_error[kc]))
                bias[i, j, ip] = mean(d.ws10_error[kc])
                
                grp = groupby(d[kc,:], :site)
                sde[i, j, ip] = mean(combine(grp, :ws10_error => std).ws10_error_std)
                sd[i, j, ip]  = mean(combine(grp, :ws10 => std).ws10_std ./
                                     combine(grp, :obs_ws10 => std).obs_ws10_std)
                dmax[i, j, ip] = mean(combine(grp, :ws10 => maximum).ws10_maximum .-
                                      combine(grp, :obs_ws10 => maximum).obs_ws10_maximum)
                rmax[i, j, ip] = mean(combine(grp, :ws10 => maximum).ws10_maximum ./
                                      combine(grp, :obs_ws10 => maximum).obs_ws10_maximum)
                if m in ["meps", "ens"]
                    mbr = m == "ens" ? (0:50) : (0:14)
                    k   = Symbol.("ws10_", mbr)
                    ens = Matrix(d[kc, k])
                    crps[i, j, ip] = crps_ensemble(ens, d.obs_ws10[kc])
                    sharp[i, j, ip] = mean(std(ens, dims=1))
                    sharp_sd[i, j, ip] = std(std(ens, dims=1))
                end
            end
        else
            for (i, l) in enumerate(lts)
                kc = d.lt .== l
                mae[i, j, ip]  = mean(abs.(d.t2_error[kc]))
                bias[i, j, ip] = mean(d.t2_error[kc])
                grp = groupby(d[kc,:], :site)
                sde[i, j, ip] = mean(combine(grp, :t2_error => std).t2_error_std)
                sd[i, j, ip]  = mean(combine(grp, :t2 => std).t2_std ./
                                     combine(grp, :obs_t2 => std).obs_t2_std)
                dmax[i, j, ip] = mean(combine(grp, :t2 => maximum).t2_maximum .- 273.15f0 .-
                                      combine(grp, :obs_t2 => maximum).obs_t2_maximum)
                dmin[i, j, ip] = mean(combine(grp, :t2 => minimum).t2_minimum .- 273.15f0 .-
                                      combine(grp, :obs_t2 => minimum).obs_t2_minimum)
                rmax[i, j, ip] = mean((combine(grp, :t2 => maximum).t2_maximum .- 273.15f0) ./
                                      combine(grp, :obs_t2 => maximum).obs_t2_maximum)
                if m in ["meps", "ens"]
                    mbr = m == "ens" ? (0:50) : (0:14)
                    k   = Symbol.("t2_", mbr)
                    ens = Matrix(d[kc,k]) .- 273.15f0
                    crps[i, j, ip] = crps_ensemble(ens, d.obs_t2[kc])
                    sharp[i, j, ip] = mean(std(ens, dims=1))
                    sharp_sd[i, j, ip] = std(std(ens, dims=1))
                end
            end          
        end
    end
    
end

scores_raw = (mae = mae, bias = bias, sde = sde, sd = sd, dmax = dmax, dmin = dmin, rmax = rmax, crps = crps,
              sharpness = sharp, sharpness_sd = sharp_sd,
              lts = lts, models = models, prm = prm)
scores_raw.crps
scores_raw.mae
scores_raw.bias
scores_raw.sde
JLD2.@save "./data/scores_raw_+$(max_lt).jld2" scores_raw

