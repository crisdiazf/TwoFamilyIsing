using TwoFamilyIsing

# Same as make_example_dataset.jl in every respect except n_links (20 -> 100),
# to isolate cross-family link *density* as the one variable under test:
# 20/900 ≈ 2.2% cross density vs ~12% intra-family density before; 100/900 ≈
# 11.1% brings cross density roughly on par with intra-family density.
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
    (-0.05, 0.05)          # field_range
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
    (-0.05, 0.05)          # field_range
)

cross = CrossSpec(
    :sparse_pairs,         # structure
    100,                   # n_links (was 20)
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
save_dataset("data/dataset_100links_09strength.bin", data)
println("Saved dataset to data/dataset_100links_09strength.bin")
