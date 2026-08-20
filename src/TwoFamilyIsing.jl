module TwoFamilyIsing

using Random
using LinearAlgebra
using Statistics
using Serialization

include("types.jl")
include("builders.jl")
include("sampling.jl")
include("stats.jl")
include("datasets.jl")
include("io.jl")

export FamilySpec,
       CrossSpec,
       ModelSpec,
       IsingModel,
       IsingDataset,
       build_family,
       build_cross,
       build_model,
       energy,
       delta_energy_flip_A,
       delta_energy_flip_B,
       metropolis_sweep!,
       sample_joint,
       empirical_means,
       empirical_corr,
       empirical_cross_corr,
       summarize_dataset,
       make_dataset,
       save_dataset,
       load_dataset

end
