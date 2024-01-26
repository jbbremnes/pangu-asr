using Statistics


"""
    qtloss(qt, y, prob; agg = mean)

computes the composite quantile loss for the levels given by `prob`.    
```
 qt    quantiles of size (prob, samples)
 y     observations
 prob  quantile levels/probabilities
 agg   aggregation function with `mean` as default. Other useful functions are `identity`,
       `u -> sum(w .* u) / sum(w)` (weighted mean), ...
```
"""
function qtloss(qt::AbstractMatrix, y::AbstractVector, prob::AbstractVector; agg = mean)
    err = y .- qt'
    return agg( (prob' .- (err .< 0)) .* err )
end


"""
    crps_ensemble(ens, y; agg = mean)

computes sample based continous ranked probability score
```
 ens   ensemble of size (samples, members)
 y     observations
 agg   aggregation function with `mean` as default. 
"""
function crps_ensemble(ens::AbstractMatrix, y::AbstractVector; agg = mean)

    crps = Vector{eltype(ens)}(undef, size(ens,1))
    m    = size(ens, 2)
    
    for i in axes(ens,1)
        crps1 = zero(eltype(ens))
        for j in axes(ens,2)
            crps1 += abs(ens[i,j] - y[i])
        end
        crps2 = zero(eltype(ens))
        for j in axes(ens,2)
            for k in axes(ens,2)
                crps2 += abs(ens[i,j] - ens[i,k])
            end
        end
        crps[i] = crps1/m - crps2/(2*m*m)
    end
    
    return agg(crps)
end

