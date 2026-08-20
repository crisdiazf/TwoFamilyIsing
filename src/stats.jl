function empirical_means(X::AbstractMatrix{<:Real})
    return vec(mean(X; dims=1))
end

function empirical_corr(X::AbstractMatrix{<:Real})
    n = size(X, 1)
    μ = empirical_means(X)
    Xc = X .- reshape(μ, 1, :)
    return (Xc' * Xc) / n
end

function empirical_cross_corr(XA::AbstractMatrix{<:Real}, XB::AbstractMatrix{<:Real})
    n = size(XA, 1)
    @assert size(XB, 1) == n
    μA = empirical_means(XA)
    μB = empirical_means(XB)
    XAc = XA .- reshape(μA, 1, :)
    XBc = XB .- reshape(μB, 1, :)
    return (XAc' * XBc) / n
end

function summarize_dataset(data::IsingDataset)
    XA = data.XA_train
    XB = data.XB_train
    X = data.X_joint_train

    CA = empirical_corr(XA)
    CB = empirical_corr(XB)
    CAB = empirical_cross_corr(XA, XB)

    return Dict(
        :n_train => size(X, 1),
        :nA => size(XA, 2),
        :nB => size(XB, 2),
        :mean_abs_JA => mean(abs.(data.model.JA)),
        :mean_abs_JB => mean(abs.(data.model.JB)),
        :mean_abs_K => mean(abs.(data.model.K)),
        :nnz_JA => count(!iszero, data.model.JA) ÷ 2,
        :nnz_JB => count(!iszero, data.model.JB) ÷ 2,
        :nnz_K => count(!iszero, data.model.K),
        :mean_abs_corr_A => mean(abs.(CA)),
        :mean_abs_corr_B => mean(abs.(CB)),
        :mean_abs_cross_corr => mean(abs.(CAB)),
    )
end
