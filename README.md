# pangu-asr
Post-processing of Pangu-Weather and NWP models.


# Installation
* clone the repository
* change directory to pangu-asr
* download the JLD2 file and store it under the ./data directory (can be a symbolic link)
* julia --threads=auto --project=./ 'using Pkg; Pkg.instantiate()'


# Training and validation 


```
julia --threads=auto --project=./ t2 60 pangu
julia --threads=auto --project=./ t2 60 hres
julia --threads=auto --project=./ t2 60 ens
julia --threads=auto --project=./ t2 60 meps
julia --threads=auto --project=./ t2 60 ens0
julia --threads=auto --project=./ t2 60 meps0
julia --threads=auto --project=./ ws10 60 pangu
julia --threads=auto --project=./ ws10 60 hres
julia --threads=auto --project=./ ws10 60 ens
julia --threads=auto --project=./ ws10 60 meps
julia --threads=auto --project=./ ws10 60 ens0
julia --threads=auto --project=./ ws10 60 meps0
```

