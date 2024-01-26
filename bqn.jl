#  Bernstein Quantile Networks with and without censoring

using Flux
using Statistics, Dates
using Printf: @sprintf
using LinearAlgebra: tril
using Interpolations: LinearInterpolation, ConstantInterpolation, Flat


include("loss_functions.jl")


timeit(tm::DateTime) = Dates.toms(Dates.now() - tm) / 1000



"""
    BQNmodel

Struct for information related to the training and prediction of BQNet models.

# Elements
- `model`: Flux model
- `degree::UInt`: the degree of the Bernstein polynomials
- `increments::Bool`: if true, the output of the `model` is treated as increments
- `censored_left`: left censored value. Use -Inf for no censoring
- `training_prob::AbstractVector`: the quantile levels applied in the loss function during training
- `training_loss::AbstractVector`: the quantile loss on the training data for each epoch
- `validation_loss::AbstractVector`: the quantile loss on the validation data for each epoch
- `total_time::Real`: total training time
- `training_time::AbstractVector`: duration of the training step for each epoch
- `validation_time::AbstractVector`: duration of the validation step for each epoch
- `learning_rates::AbstractVector`: the learning rate for each epoch
- `epochs::UInt`: the total number of training epochs
"""
struct BQNmodel
    model
    degree::Integer
    increments::Bool
    censored_left::Real
    training_prob::AbstractVector
    training_loss::AbstractVector
    validation_loss::AbstractVector
    total_time::Real
    training_time::AbstractVector
    validation_time::AbstractVector
    learning_rates::AbstractVector
    epochs::Integer
end

import Base.show
function show(io::IO, fit::BQNmodel)
    println(io, "Degree of Bernstein basis polynomials: ", fit.degree)
    println(io, "Non-decreasing Bernstein basis: ", fit.increments)
    println(io, "Left censored value: ", fit.censored_left)
    println(io, "Network: ", fit.model)
    println(io, "Network/model parameters: ", sum(length, Flux.params(fit.model)))
    trloss = findmin(fit.training_loss)
    println(io, "Best training loss: ", Float32(trloss[1]), " (epoch ", trloss[2], 
            ", learning rate ", fit.learning_rates[trloss[2]], ")")
    valloss = findmin(fit.validation_loss)
    println(io, "Best validation loss: ", Float32(valloss[1]), " (epoch ", valloss[2],
            ", learning rate ", fit.learning_rates[valloss[2]], ")")
    println(io, "Total training time: ", fit.total_time, " sec")
    println(io, "Average training time per epoch: ", Float16.(mean(fit.training_time)), " sec")
    println(io, "Average validation time per epoch: ", Float16.(mean(fit.validation_time)), " sec")
    println(io, "Number of epochs: ", fit.epochs)
    println(io, "Initial learning rate: ", fit.learning_rates[1])
    println(io, "End learning rate: ", fit.learning_rates[fit.epochs])
end

"""
    BernsteinMatrix(degree::Integer, prob::AbstractVector)

Compute the Bernstein basis polynomials of degree `degree` at `prob`. A matrix
of size (levels, degree+1) is returned.
"""
BernsteinMatrix(degree::Integer, prob::AbstractVector) =
    [binomial(degree, d) * p^d * (1-p)^(degree-d) for p in prob, d in 0:degree]
    



#  function for timing
timeit(tm::DateTime) = Dates.toms(Dates.now() - tm) / 1000

#  activation function for increments
softplus_increment(x::Matrix) = vcat(x[1:1, :], softplus(x[2:end, :]))


#  training loop for censored Bernstein Quantile Networks
function bqn_train!(model, tr_loader, val_loader;
                    increments::Bool = true,
                    prob::AbstractVector = Float32.(0:0.01:1),
                    censored_left = NaN, censored_prob = false,
                    learning_rate::AbstractFloat = 0.001,
                    learning_rate_scale::AbstractFloat = 0.1,
                    learning_rate_min::AbstractFloat = 5e-6,
                    patience::Integer = 10, max_epochs::Integer = 200,
                    best_model::Bool = true, device::Function = cpu)
    
    prob_tr    = Float32.(prob)
    censored   = isfinite(censored_left) ? true : false
    prob_cens  = censored ? fill(Float32(mean(tr_loader.data[2] .<= censored_left)), tr_loader.batchsize) : 0f0
    degree     = size(model(first(tr_loader)[1]))[end-1] - 1
    B          = BernsteinMatrix(degree, prob_tr)
    if increments
        B = B * tril(ones(Float32, degree+1, degree+1)) 
    end
    B          = B |> device
    prob_tr    = prob_tr |> device
    mask       = ones(Float32, degree+1, tr_loader.batchsize) |> device   # fixed batchsize assumed!
    agg        = censored ? u -> sum(u .* mask) / sum(mask)  :  mean
    loss(x, y) = qtloss(B * model(x), y, prob_tr; agg = agg)                     
    prm        = Flux.params(model) 
    opt        = ADAM(learning_rate)
    qs_tr      = Float64[]  
    qs_val     = Float64[]  
    lrs        = Float64[]
    tm_tr      = Float64[]
    tm_val     = Float64[]
    masked     = zeros(Float64, max_epochs)
    prob_inv   = 1f0 .- transpose(prob_tr)
    bmodel     = deepcopy(model)
    ictr       = 1
    epochs     = 0

    tm_total   = Dates.now()
    for i in 1:max_epochs

        epochs += 1
        push!(lrs, opt.eta)
        tm  = Dates.now()
        local trloss = 0f0
        for (x, y) in tr_loader
            x = x |> device
            y = y |> device
            if censored
                if i == 1 && censored_prob
                    mask = Float32.(prob_cens .> prob_inv)
                else
                    mask = Float32.((B * model(x))' .> censored_left)
                end                
                masked[i] += sum(mask)
            end
            gs = gradient(prm) do
                trloss += loss(x, y)
                return trloss
            end
            Flux.update!(opt, prm, gs)
        end
        push!(qs_tr, trloss / length(tr_loader))
        push!(tm_tr, timeit(tm))
        masked[i] = masked[i] / length(tr_loader)  # length(tr_loader.data)
        
        tm      = Dates.now()
        valloss = 0.0
        for (x, y) in val_loader
            x = x |> device
            y = y |> device
            qt = censored ? max.(censored_left, B*model(x)) : B*model(x)
            valloss += qtloss(qt, y, prob_tr) 
        end
        push!(qs_val, valloss / length(val_loader))
        push!(tm_val, timeit(tm))
        
        lr        = ""
        if findmin(qs_val)[2] == i   # last fit the best?
            if best_model
                bmodel = deepcopy(model)
            end
            ictr = 1
        else
            if ictr > patience
                opt.eta *= learning_rate_scale
                lr       = string(opt.eta)
                ictr     = 1
            else
                ictr    += 1
            end
        end
 
        @info @sprintf("%4d: quantile scores training and validation: %.5f  %.5f %s  %.3fs/%.3fs %s",
                       i, qs_tr[i], qs_val[i], i == findmin(qs_val)[2] ? "*" : " ",
                       tm_tr[i], tm_val[i],
                       lr == "" ? lr : string("\n      new learning rate: ", lr))
        if opt.eta < learning_rate_min
            break
        end
    end
    tm_total = timeit(tm_total)
    
    println("best quantile validation score: ", findmin(qs_val)) 
    model_output = best_model ? deepcopy(bmodel) : deepcopy(model)
    model_output = model_output |> cpu
    prob_tr = prob_tr |> cpu
   
    return BQNmodel(model_output, degree, increments, censored_left, prob_tr,
                    qs_tr, qs_val, tm_total, tm_tr, tm_val, lrs, epochs)
end



"""
    predict(fit::BQNmodel, x; prob::AbstractVector = fit.training_prob)

Compute conditional quantiles for levels `prob` at `x` based on the BQN model `fit`.
"""
function predict(fit::BQNmodel, x; prob::AbstractVector = fit.training_prob)
    B = BernsteinMatrix(fit.degree, Float32.(prob))
    if fit.increments
        B = B * tril(ones(eltype(B), fit.degree+1, fit.degree+1)) 
    end
    return isfinite(fit.censored_left) ? max.(fit.censored_left, B * fit.model(x)) : B * fit.model(x)
end


"""
   cdf(fit::BQNmodel, x, y::AbstractVector; prob::AbstractVector = Float32.(0:0.01:1))

Compute the conditional cumulative distribution function of Y|x for values `y` for each `x`
based on the BQN model `fit`. The CDFs evaluated at `y` are simply obtained by computing the
proportion of predicted quantiles less or equal to `y`. The size of `prob` determines the
accuracy of the approximation.
"""
function cdf(fit::BQNmodel, x, y::AbstractVector;
             prob::AbstractVector = Float32.(0:0.0025:1))
    n   = isa(x, Tuple) ? size(x[end])[end] : size(x)[end]
    p   = Float32.(prob)
    yy  = Float32.(y)
    qts = predict(fit, x, prob = p)
    out = zeros(Float32, n, length(y))
    for i in eachindex(y)
        out[:, i] = mean(qts .<= yy[i], dims = 1)[:]
    end
    return out
end

