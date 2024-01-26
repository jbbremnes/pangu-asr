#  train separate BQN models for t2m and ws10 for various NWP and Pangu models
#
#  usage: julia --threads=auto --project=./ train.jl <prm> <max_lt> <source model>
#
#  examples:  julia --threads=auto --project=./ train.jl t2 60 pangu
#             julia --threads=auto --project=./ train.jl ws10 60 pangu
#


using JLD2, CSV, DataFrames, Statistics, Random
using Flux, BSON

include("bqn.jl")


#  get data
function getdata(; model = "pangu", prm = "t2", max_lt = 60)

    println("model: $model\nparameter: $prm\nmaximum lead time: $max_lt")
    files = string("./data/", model, "+obs.csv")
       
    #  read data
    JLD2.@load "./data/nwp+obs.jld2"
    df = data[indexin([model], models)][1]
    filter!(row -> row.lt <= max_lt, df)
    
    #  convert to float32
    k = map(u -> eltype(u) == Float64, eachcol(df))
    df[!, k] = Float32.(df[:, k])
    df.lt    = Float32.(df.lt)
    
    #  create additional predictors
    site    = string.(unique(df.site))
    df.sid  = Int.(indexin(df.site, site))
    df.yday = dayofyear.(DateTime.(df.time_ref))
    df.yday_cos = Float32.(cos.(df.yday/365*2pi))
    df.yday_sin = Float32.(sin.(df.yday/365*2pi))
    
    #  define target and input variables
    yvar = prm == "t2" ? :obs_t2 : :obs_ws10
    if model in ["ens", "meps"]
        if prm == "t2"
            xvar = [:sid, :lt, :yday_cos, :yday_sin, :t2_mean, :t2_sd]
        else
            xvar = [:sid, :lt, :yday_cos, :yday_sin, :u10_mean, :v10_mean, :ws10_mean, :ws10_sd]
        end
    else
        if prm == "t2"
            xvar = [:sid, :lt, :yday_cos, :yday_sin, :t2]
        else
            xvar = [:sid, :lt, :yday_cos, :yday_sin, :u10, :v10, :ws10]
        end
    end
    println("input variables: ", xvar)
    
    #  tidy data
    df = df[:, [:site; :time; :time_ref; xvar; yvar]]
    df = df[completecases(df), :]

    #  standardise and split into training, validation and test datasets
    tm = DateTime.(df.time)
    tm_ref = DateTime.(df.time_ref)
    vday = day.(tm_ref) .> 25
    k  =  (training = (year.(tm_ref) .== 2021) .&& .!vday,
           validation = (year.(tm_ref) .== 2021) .&& vday,
           test = year.(tm_ref) .== 2022)
    m  = mean(Matrix(df[k.training, xvar]), dims=1)
    s  = std(Matrix(df[k.training, xvar]), dims=1)
    xx = (Matrix(df[:, xvar]) .- m) ./ s
    println("size of training, validation and test datasets: ",
            sum(k.training), ", ", sum(k.validation), ", ", sum(k.test))
    
    return (training = (x = (df[k.training,:sid], xx[k.training,:]'), y = df[k.training, yvar]),
            validation = (x = (df[k.validation,:sid], xx[k.validation,:]'), y = df[k.validation, yvar]),
            test = (x = (df[k.test,:sid], xx[k.test,:]'), y = df[k.test, yvar]),
            test_meta = df[k.test, [:site, :time, :time_ref, :lt]],
            xvar = xvar, yvar = yvar, sites = length(unique(df.sid)))
end




function main()

    prm = ARGS[1]                    # parameter: t2 or ws10
    max_lt = parse(Int, ARGS[2])     # maximum lead time: 60
    src = ARGS[3]                    # model: pangu, hres, ens, meps, ens0, meps0
   
    #  get and organise data
    data = getdata(; model = src, prm = prm, max_lt = max_lt)
       
    #  create data loaders
    train_loader = Flux.Data.DataLoader(data.training, batchsize = 128, shuffle = true, partial = false)
    val_loader = Flux.Data.DataLoader(data.validation, batchsize = size(data.validation.y)[end], shuffle = false)

    #  train and predict
    units  = [ [64,32], [32,32], [32,16] ]
    degree = 12
    emb    = data.sites
    for b in 1:3

        println("Bootstrap $b")

        for m in eachindex(units)
  
            #  create model
            println("Model $m")
            print("  create model ... ")
            model = Chain(Parallel(vcat, Embedding(emb, 8), identity),
                          Dense(8+length(data.xvar), units[m][1], elu),
                          Dense(units[m][1], units[m][2], elu),
                          Dense(units[m][2], degree+1, identity), softplus_increment)
            println("done")
        
            #  train model
            println("  train model ... ")
            @time fit = bqn_train!(model, train_loader, val_loader;
                                   increments = true,
                                   prob = Float32.(0.025:0.05:0.975), 
                                   device = cpu)
            BSON.@save "./data/bqn_model_$(prm)_+$(max_lt)_$(src)_m$(m)b$(b).bson" fit
            println("done")
        
            #  make prediction
            print("  making predictions ... ")
            prob_qts = Float32.(0:0.01:1)
            qts = predict(fit, data.test.x; prob = prob_qts)
            qts = DataFrame(qts', string.("Q", round.(Int, prob_qts .* 100)))
            out = hcat(data.test_meta, DataFrame(obs = data.test.y), qts)
            println("done")

            #  write predictions to csv file
            print("  write to file ... ")
            CSV.write("./data/qts_bqn_$(prm)_+$(max_lt)_$(src)_m$(m)b$(b).csv", out)
            println("done")
        end
        
    end
    
end



@time main()

