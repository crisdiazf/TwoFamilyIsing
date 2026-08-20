function energy(model::IsingModel, sA::AbstractVector{<:Integer}, sB::AbstractVector{<:Integer})
    JA, JB, K = model.JA, model.JB, model.K
    hA, hB = model.hA, model.hB

    eA = -0.5 * dot(sA, JA * sA) - dot(hA, sA)
    eB = -0.5 * dot(sB, JB * sB) - dot(hB, sB)
    eAB = -dot(sA, K * sB)

    return eA + eB + eAB
end

function delta_energy_flip_A(model::IsingModel, sA::Vector{Int8}, sB::Vector{Int8}, i::Int)
    local_field = dot(model.JA[i, :], sA) + dot(model.K[i, :], sB) + model.hA[i]
    return 2.0 * sA[i] * local_field
end

function delta_energy_flip_B(model::IsingModel, sA::Vector{Int8}, sB::Vector{Int8}, j::Int)
    local_field = dot(model.JB[j, :], sB) + dot(model.K[:, j], sA) + model.hB[j]
    return 2.0 * sB[j] * local_field
end

function metropolis_sweep!(model::IsingModel, sA::Vector{Int8}, sB::Vector{Int8}, rng::AbstractRNG)
    nA = length(sA)
    nB = length(sB)

    for i in randperm(rng, nA)
        dE = delta_energy_flip_A(model, sA, sB, i)
        if dE <= 0 || rand(rng) < exp(-model.beta * dE)
            sA[i] = Int8(-sA[i])
        end
    end

    for j in randperm(rng, nB)
        dE = delta_energy_flip_B(model, sA, sB, j)
        if dE <= 0 || rand(rng) < exp(-model.beta * dE)
            sB[j] = Int8(-sB[j])
        end
    end

    return nothing
end

function sample_joint(model::IsingModel;
    n_samples::Int,
    burnin::Int=5_000,
    thinning::Int=20,
    seed::Int=1234,
    init_mode::Symbol=:random
)
    rng = MersenneTwister(seed)

    nA = size(model.JA, 1)
    nB = size(model.JB, 1)

    if init_mode == :random
        sA = Int8.(rand(rng, Bool, nA) .* 2 .- 1)
        sB = Int8.(rand(rng, Bool, nB) .* 2 .- 1)
    elseif init_mode == :plus
        sA = fill(Int8(1), nA)
        sB = fill(Int8(1), nB)
    elseif init_mode == :minus
        sA = fill(Int8(-1), nA)
        sB = fill(Int8(-1), nB)
    else
        error("unknown init_mode: $init_mode")
    end

    for _ in 1:burnin
        metropolis_sweep!(model, sA, sB, rng)
    end

    X = Matrix{Int8}(undef, n_samples, nA + nB)

    for t in 1:n_samples
        for _ in 1:thinning
            metropolis_sweep!(model, sA, sB, rng)
        end
        X[t, 1:nA] = sA
        X[t, nA+1:nA+nB] = sB
    end

    return X
end
