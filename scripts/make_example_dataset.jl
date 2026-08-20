using TwoFamilyIsing

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
    20,                    # n_links
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

"""
XA = data.XA_train
XB = data.XB_train
X = data.X_joint_train

CA = empirical_corr(XA)
CB = empirical_corr(XB)
CAB = empirical_cross_corr(XA, XB)

heatmap(data.model.JA)
heatmap(CA)

heatmap(data.model.JB)
heatmap(CB)

heatmap(data.model.K)
heatmap(CAB)
"""

mkpath("data")
save_dataset("data/dataset_20links_09strength.bin", data)
println("Saved dataset to data/example_dataset.bin")
