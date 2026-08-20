function _split_counts(n_total::Int, train_frac::Float64, val_frac::Float64)
    @assert 0 < train_frac < 1
    @assert 0 <= val_frac < 1
    @assert train_frac + val_frac < 1

    n_train = floor(Int, train_frac * n_total)
    n_val = floor(Int, val_frac * n_total)
    n_test = n_total - n_train - n_val

    @assert n_train > 0 && n_val >= 0 && n_test > 0
    return n_train, n_val, n_test
end

function make_dataset(spec::ModelSpec;
    n_total_samples::Int=10_000,
    train_frac::Float64=0.7,
    val_frac::Float64=0.15,
    burnin::Int=5_000,
    thinning::Int=20,
    sample_seed::Int=5678
)
    model = build_model(spec)
    X = sample_joint(
        model;
        n_samples=n_total_samples,
        burnin=burnin,
        thinning=thinning,
        seed=sample_seed
    )

    nA = spec.familyA.n
    nB = spec.familyB.n
    @assert size(X, 2) == nA + nB

    n_train, n_val, n_test = _split_counts(n_total_samples, train_frac, val_frac)

    X_joint_train = X[1:n_train, :]
    X_joint_val = X[n_train+1:n_train+n_val, :]
    X_joint_test = X[n_train+n_val+1:end, :]

    XA_train = X_joint_train[:, 1:nA]
    XA_val = X_joint_val[:, 1:nA]
    XA_test = X_joint_test[:, 1:nA]

    XB_train = X_joint_train[:, nA+1:nA+nB]
    XB_val = X_joint_val[:, nA+1:nA+nB]
    XB_test = X_joint_test[:, nA+1:nA+nB]

    metadata = Dict{Symbol,Any}(
        :n_total_samples => n_total_samples,
        :burnin => burnin,
        :thinning => thinning,
        :sample_seed => sample_seed
    )

    return IsingDataset(
        spec,
        model,
        X_joint_train,
        X_joint_val,
        X_joint_test,
        XA_train,
        XA_val,
        XA_test,
        XB_train,
        XB_val,
        XB_test,
        metadata
    )
end
