function _rand_weight(rng::AbstractRNG, wrange::Tuple{Float64,Float64}; allow_negative::Bool=true)
    lo, hi = wrange
    @assert lo <= hi "invalid weight range"
    if allow_negative
        w = rand(rng) * (hi - lo) + lo
        while abs(w) < 1e-12
            w = rand(rng) * (hi - lo) + lo
        end
        return w
    else
        lo2 = max(lo, 1e-8)
        hi2 = max(hi, lo2 + 1e-8)
        return rand(rng) * (hi2 - lo2) + lo2
    end
end

function _sample_fields(rng::AbstractRNG, n::Int, frange::Tuple{Float64,Float64})
    lo, hi = frange
    return rand(rng, n) .* (hi - lo) .+ lo
end

function _make_modules(module_sizes::Vector{Int})
    modules = Int[]
    for (m, sz) in enumerate(module_sizes)
        append!(modules, fill(m, sz))
    end
    return modules
end

function _check_family_spec(spec::FamilySpec)
    @assert spec.n > 0 "family size must be positive"
    if spec.structure == :modular
        @assert sum(spec.module_sizes) == spec.n "sum(module_sizes) must equal n"
    elseif spec.structure == :sparse_random
        @assert 0.0 <= spec.edge_density <= 1.0 "edge_density must be in [0,1]"
    else
        error("unknown family structure: $(spec.structure)")
    end
end

function _check_cross_spec(spec::CrossSpec, nA::Int, nB::Int)
    @assert spec.n_links >= 0 "n_links must be nonnegative"
    @assert nA > 0 && nB > 0
    if spec.structure ∉ (:sparse_pairs, :sparse_block)
        error("unknown cross structure: $(spec.structure)")
    end
end

function build_family(spec::FamilySpec, rng::AbstractRNG)
    _check_family_spec(spec)

    J = zeros(Float64, spec.n, spec.n)
    h = _sample_fields(rng, spec.n, spec.field_range)
    modules = zeros(Int, spec.n)

    if spec.structure == :modular
        modules = _make_modules(spec.module_sizes)

        for i in 1:spec.n-1
            for j in i+1:spec.n
                same_module = modules[i] == modules[j]
                if same_module
                    if rand(rng) < spec.p_in
                        w = _rand_weight(rng, spec.J_in_range; allow_negative=false)
                        J[i, j] = w
                        J[j, i] = w
                    end
                else
                    if rand(rng) < spec.p_out
                        w = _rand_weight(rng, spec.J_out_range; allow_negative=false)
                        J[i, j] = w
                        J[j, i] = w
                    end
                end
            end
        end

    elseif spec.structure == :sparse_random
        modules .= 1
        for i in 1:spec.n-1
            for j in i+1:spec.n
                if rand(rng) < spec.edge_density
                    w = _rand_weight(rng, spec.J_range; allow_negative=spec.allow_negative)
                    J[i, j] = w
                    J[j, i] = w
                end
            end
        end
    end

    return J, h, modules
end

function build_cross(spec::CrossSpec, modulesA::Vector{Int}, modulesB::Vector{Int}, rng::AbstractRNG)
    nA = length(modulesA)
    nB = length(modulesB)
    _check_cross_spec(spec, nA, nB)

    K = zeros(Float64, nA, nB)
    edges = Tuple{Int,Int,Float64}[]

    candidate_pairs = Tuple{Int,Int}[]

    if spec.structure == :sparse_pairs
        for i in 1:nA, j in 1:nB
            push!(candidate_pairs, (i, j))
        end

    elseif spec.structure == :sparse_block
        allowedA = isempty(spec.a_modules) ? unique(modulesA) : spec.a_modules
        allowedB = isempty(spec.b_modules) ? unique(modulesB) : spec.b_modules

        for i in 1:nA, j in 1:nB
            if (modulesA[i] in allowedA) && (modulesB[j] in allowedB)
                push!(candidate_pairs, (i, j))
            end
        end
    end

    @assert spec.n_links <= length(candidate_pairs) "n_links exceeds available cross pairs"

    perm = randperm(rng, length(candidate_pairs))
    chosen = candidate_pairs[perm[1:spec.n_links]]

    for (i, j) in chosen
        w = _rand_weight(rng, spec.K_range; allow_negative=spec.allow_negative)
        K[i, j] = w
        push!(edges, (i, j, w))
    end

    return K, edges
end

function build_model(spec::ModelSpec)
    rng = MersenneTwister(spec.seed)

    JA, hA, modulesA = build_family(spec.familyA, rng)
    JB, hB, modulesB = build_family(spec.familyB, rng)
    K, edges = build_cross(spec.cross, modulesA, modulesB, rng)

    return IsingModel(JA, JB, K, hA, hB, spec.beta, modulesA, modulesB, edges)
end
