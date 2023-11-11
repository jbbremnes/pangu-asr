## Post-processing of Pangu-Weather and NWP models.
The repository contains Julia source code for the article [Evaluation of forecasts by a global data-driven weather model with and without probabilistic post-processing at Norwegian stations](https://arxiv.org/abs/2309.01247).

## Installation
To setup the environment clone the repository, instatiate the julia project and download the data
```
git clone git@github.com:jbbremnes/pangu-asr.git
cd pangu-asr
mkdir data         # could also be symbolic link to a directory for the data
mkdir data/plots
julia --project=./ 'using Pkg; Pkg.instantiate()'

```

* clone the repository
* change directory to pangu-asr
* download the JLD2 file and store it under the ./data directory (can be a symbolic link)
* julia --threads=auto --project=./ 'using Pkg; Pkg.instantiate()'

##  Data
The forecast and observational data is stored in a single JLD2 file and can be read in Julia by
```
julia> using JLD2, DataFrames

julia> JLD2.@load "data/nwp+obs.jld2"
2-element Vector{Symbol}:
 :data                      # vector of 6 data frames each with 1_223_264 cases
 :models                    # names of the 6 models

julia> models
6-element Vector{String}:
 "pangu"
 "hres"
 "ens"
 "meps"
 "ens0"
 "meps0"
```


## Training and forecast validation
Train models for temperature (2m) and wind speed (10m) for each of the six forecast models by
```
julia --project=./ train.jl t2 60 pangu
julia --project=./ train.jl t2 60 hres
julia --project=./ train.jl t2 60 ens
julia --project=./ train.jl t2 60 meps
julia --project=./ train.jl t2 60 ens0
julia --project=./ train.jl t2 60 meps0
julia --project=./ train.jl ws10 60 pangu
julia --project=./ train.jl ws10 60 hres
julia --project=./ train.jl ws10 60 ens
julia --project=./ train.jl ws10 60 meps
julia --project=./ train.jl ws10 60 ens0
julia --project=./ train.jl ws10 60 meps0
```
from the `pangu-asr` directory. 3×3 BQN models are trained for each parameter/forecast model combination. The training takes around 4 hours in total.

Verification statistics can be computed by
```
julia --threads=auto --project=./ verification.jl
```
and plots by
```
julia --threads=auto --project=./ plots.jl
```

