using TwoFamilyIsing

# Same as make_dataset_100links.jl in every respect except field_range
# (±0.05 -> ±0.3), to fix the "too-weak <v> signal" problem: with the
# original narrow field range, the true site-to-site variation in <v> has
# std ≈ 0.008, comparable to or smaller than realistic sampling noise floors
# (and, for the paired model specifically, smaller than the ~0.015 collateral
# marginal shift imparted by the added hidden units as a side effect of
# encoding real cross-family correlation — see the <v> alignment discussion).
# Widening the field range to ±0.3 raises the true signal to std ≈ 0.04-0.05,
# comfortably above both noise floors, so the <v> data-vs-model scatter
# aligns cleanly at the same sample sizes used everywhere else in the
# pipeline, without needing a special high-N_SAMPLES run just for that one
# panel. J_range/K_range/edge_density/n_links are left untouched so the
# pairwise (intra- and cross-family) correlation structure — the actual
# object of interest — keeps the same relative importance as before.
familyA = FamilySpec(
    30,                    # n
    :sparse_random,        # structure
    Int[],                 # module_sizes
    0.0,                   # p_in
    0.0,                   # p_out
    (0.0, 0.0),            # J_in_range
    (0.0, 0.0),            # J_out_range
    0.12,                  # edge_density
    (-0.8, 0.8),           # J_range
    true,                  # allow_negative
    (-0.3, 0.3)            # field_range (was (-0.05, 0.05))
)


familyB = FamilySpec(
    30,                    # n
    :sparse_random,        # structure
    Int[],                 # module_sizes
    0.0,                   # p_in
    0.0,                   # p_out
    (0.0, 0.0),            # J_in_range
    (0.0, 0.0),            # J_out_range
    0.12,                  # edge_density
    (-0.8, 0.8),           # J_range
    true,                  # allow_negative
    (-0.3, 0.3)            # field_range (was (-0.05, 0.05))
)

cross = CrossSpec(
    :sparse_pairs,         # structure
    100,                   # n_links
    (-0.9, 0.9),           # K_range
    true,                  # allow_negative
    Int[],                 # a_modules
    Int[]                  # b_modules
)

spec = ModelSpec(
    familyA,
    familyB,
    cross,
    0.5,                   # beta
    1234                   # seed
)

@time data = make_dataset(
    spec;
    n_total_samples=20000,
    train_frac=0.7,
    val_frac=0.15,
    burnin=5000,
    thinning=100,
    sample_seed=999
)

mkpath("data")
save_dataset("data/dataset_100links_09strength_strongfield.bin", data)
println("Saved dataset to data/dataset_100links_09strength_strongfield.bin")
