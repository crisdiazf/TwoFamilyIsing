using CUDA
using CairoMakie
using HDF5
using RestrictedBoltzmannMachines: sample_v_from_v, sample_h_from_v, sample_from_inputs
using RestrictedBoltzmannMachines: free_energy, load_rbm, gpu, cpu, Falses
using Statistics: mean, std, cor
using LinearAlgebra: diagm, diag
using Random
using OneHotArrays
using TwoFamilyIsing

Random.seed!(42)

# =============================================================================
# CONFIG
# =============================================================================
# This is the "real" validation companion to analyze_results.jl: that script
# only reads training-time logs (short PCD chains); this one loads the actual
# saved models and draws long, properly-equilibrated Gibbs chains from them,
# so the data/model comparisons and the cross-family correlation check against
# the *ground-truth* coupling matrix (data.model.K) are trustworthy rather than
# a training-time proxy. Mirrors train_potts.jl/train_binary.jl's file naming
# and analyze_results.jl's CLI convention so it can point at any trained run.
const OUTPUT_DIR = "./results"
const CD_STEPS   = 50

# Same ARGS convention as the training scripts:
# N_ITERS PAIRED_ITERS [H_ADD=30] [N_HIDDEN=30] [REG=0] [DATASET_TAG=20links_09strength] [HID_TAG=] [N_SAMPLES=1000] [SEED=42]
const N_ITERS      = parse(Int, ARGS[1])
const PAIRED_ITERS = parse(Int, ARGS[2])
const H_ADD        = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 30
const N_HIDDEN     = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 30
const REG          = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 0
const DATASET_TAG  = length(ARGS) >= 6 ? ARGS[6] : "20links_09strength"
const DATA_PATH    = "./data/dataset_$(DATASET_TAG).bin"

const N_SAMPLES     = length(ARGS) >= 8 ? parse(Int, ARGS[8]) : 1000
const N_GIBBS_STEPS = 100
const GIBBS_STRIDE  = 50   # elementary Gibbs sweeps between free-energy checkpoints

regstr(r) = isinteger(r) ? string(Int(r)) : string(r)  # REG=0 (not 0.0) matches pre-existing filenames
# Hidden-unit type for the Potts side (must match whatever tag, if any,
# train_potts.jl used when it wrote these files). Defaults to empty so this
# script keeps reading the many pre-existing nsReLU-era runs, which predate
# this tag entirely and have no "_HID=" in their filenames at all — only pass
# a 7th arg (e.g. "xReLU") when analyzing a run trained with that tag.
const HID_TAG = length(ARGS) >= 7 ? ARGS[7] : ""
hidsuffix() = isempty(HID_TAG) ? "" : "_HID=$(HID_TAG)"
potts_label = isempty(HID_TAG) ? "Potts+nsReLU" : "Potts+$(HID_TAG)"

# Same default-42/only-tag-if-different convention as train_potts.jl's SEED.
const SEED = length(ARGS) >= 9 ? parse(Int, ARGS[9]) : 42
seedsuffix() = SEED == 42 ? "" : "_SEED=$(SEED)"

# Same convention for the learning rate.
const LR = length(ARGS) >= 10 ? parse(Float64, ARGS[10]) : 1e-3
lrsuffix() = LR == 1e-3 ? "" : "_LR=$(LR)"

const ARCHS = [("" , potts_label), ("binary_", "BinaryRBM")]

# N_SAMPLES only enters the output directory name when non-default, so the
# ordinary 1000-sample validation runs keep their existing paths/figures.
nsampsuffix() = N_SAMPLES == 1000 ? "" : "_NSAMPLES=$(N_SAMPLES)"
const PARAMTAG = "N_HIDDEN=$(N_HIDDEN)_H_ADD=$(H_ADD)_N_ITERS=$(N_ITERS)_PAIRED_ITERS=$(PAIRED_ITERS)_REG=$(regstr(REG))_DATA=$(DATASET_TAG)$(hidsuffix())$(nsampsuffix())$(seedsuffix())$(lrsuffix())"
const REPORT_DIR = joinpath(OUTPUT_DIR, "pair_results_$(PARAMTAG)")
isdir(REPORT_DIR) || mkpath(REPORT_DIR)

# Every figure filename carries arch + the full param tag (not just the
# containing directory) so files don't get confused once copied elsewhere or
# viewed in a flat gallery alongside other runs.
figpath(arch, name) = joinpath(REPORT_DIR, "$(arch)$(name)_$(PARAMTAG).png")

# Private RBMs (A, B) are H_ADD-independent — train_potts.jl now saves/reuses
# them under a suffix without H_ADD, so lookups here must match.
private_suffix() = "N_HIDDEN=$(N_HIDDEN)_K=$(CD_STEPS)_N_ITERS=$(N_ITERS)_REG=$(regstr(REG))_DATA=$(DATASET_TAG)$(hidsuffix())$(seedsuffix())$(lrsuffix())"
single_suffix() = "N_HIDDEN=$(N_HIDDEN)_H_ADD=$(H_ADD)_K=$(CD_STEPS)_N_ITERS=$(N_ITERS)_REG=$(regstr(REG))_DATA=$(DATASET_TAG)$(hidsuffix())$(seedsuffix())$(lrsuffix())"
paired_suffix()  = single_suffix() * "_PAIRED_ITERS=$(PAIRED_ITERS)"
hdf5path(arch, name, suffix) = joinpath(OUTPUT_DIR, "$(arch)$(name)_$(suffix).hdf5")

# =============================================================================
# DATA
# =============================================================================
data      = load_dataset(DATA_PATH)
XA_spins  = data.XA_train'
XB_spins  = data.XB_train'
XA_states = @. Int((XA_spins + 1) / 2)
XB_states = @. Int((XB_spins + 1) / 2)
q         = maximum(XA_states) + 1
N_VIS     = size(XA_states, 1)
K_true    = data.model.K   # ground-truth A-B coupling, shape (N_VIS, N_VIS) — the actual target

function onehot_potts(states, q)
    n_sites, n_samples = size(states)
    out = falses(q, n_sites, n_samples)
    for s in 1:n_sites
        out[:, s, :] = onehotbatch(states[s, :], 0:(q-1))
    end
    return out
end

# =============================================================================
# ARCHITECTURE-GENERIC HELPERS
# =============================================================================
# Potts arrays are (q, n_vis, batch) one-hot; Binary arrays are (n_vis, batch)
# already flat. Everywhere the two need different treatment, dispatch on
# ndims rather than branching on an `arch` flag — keeps the analysis below
# identical for both architectures.
flatten(x::AbstractArray{<:Any,2}) = x
flatten(x::AbstractArray{<:Any,3}) = reshape(x, size(x, 1) * size(x, 2), size(x, 3))

vis_slice(x::AbstractArray{<:Any,2}, r) = x[r, :]
vis_slice(x::AbstractArray{<:Any,3}, r) = x[:, r, :]

occupancy_matrix(x) = flatten(x)'
visible_means(x) = vec(mean(flatten(x); dims=2))
function hv_moments(h, x)
    xf = flatten(x)
    return h * xf' / size(xf, 2)
end

# Decode a one-hot Potts array down to a single scalar state per site (i.e.
# the category index, same convention as XA_states/XB_states before
# one-hotting) — needed anywhere we compare against K_true, which is defined
# per *site*, not per (site, color) like the raw one-hot moments are. Binary
# arrays are already per-site scalars, so this is a no-op for them. Getting
# this wrong is a real bug: comparing a (2·N_VIS)×(2·N_VIS) one-hot occupancy
# correlation matrix against a (N_VIS)×(N_VIS) K_true, index-for-index, would
# silently misalign color channels with sites.
decode_state(x::AbstractArray{<:Any,2}) = Array(Float64.(x))
function decode_state(x::AbstractArray{<:Any,3})
    xc = Array(x)  # force CPU first: broadcasting a CPU `weights` against a GPU x errors
    q = size(xc, 1)
    weights = reshape(Float64.(0:q-1), q, 1, 1)
    return dropdims(sum(weights .* xc; dims=1); dims=1)
end

function empirical_cross_corr(OA, OB)
    n = size(OA, 1)
    OA_c = OA .- mean(OA, dims=1)
    OB_c = OB .- mean(OB, dims=1)
    return (OA_c' * OB_c) / (n - 1)
end

offdiag_mean_abs(M) = mean(abs.(M .- diagm(diag(M))))

# Flat vector of all off-diagonal entries (both (i,j) and (j,i) — matches the
# convention in train_potts.jl's vv_offdiag_vec) — used to scatter data-side
# vs model-side correlation *values* directly against each other, rather than
# just comparing the two heatmaps side by side or looking at a single
# summary r. This is the real point-cloud behind vv_check's r: whether it
# hugs y=x uniformly, or whether the fit is driven by a few strong points
# while the many weak/near-zero pairs scatter loosely around the origin.
offdiag_vec(M) = [M[i, j] for i in 1:size(M, 1) for j in 1:size(M, 2) if i != j]

# Long, properly-equilibrated Gibbs chain (architecture-agnostic): initialize
# from the layer's own marginal (same convention pcd! itself uses for
# persistent chains, replacing the original's `bitrand` init which produced
# invalid states for Potts and needed the first step to "fix" it), then track
# free energy over the run so burn-in can be checked visually rather than
# assumed.
function gibbs_sample(rbm, n_samples, n_steps, stride)
    x = sample_from_inputs(rbm.visible, Falses(size(rbm.visible)..., n_samples))
    F = zeros(n_samples, n_steps)
    F[:, 1] .= Array(free_energy(rbm, x))
    for t in 2:n_steps
        x = sample_v_from_v(rbm, x; steps=stride)
        F[:, t] .= Array(free_energy(rbm, x))
    end
    return x, F
end

# =============================================================================
# PLOT HELPERS (CairoMakie)
# =============================================================================
# Every panel is forced to a square box via `aspect=1`/`DataAspect()` (see
# scatter_panel!/heat_panel!/free_energy_panel! below). Figure canvas size is
# then CELL px per panel times the grid's actual (rows, cols) — a 1x3 row of
# squares gets a wide-but-short canvas rather than a forced-square canvas that
# would otherwise leave large gaps. Heatmap panels get extra column width
# (CELL_HEAT) since heat_panel! packs a colorbar into the same cell.
const CELL = 480
const CELL_HEAT = CELL + 90

function scatter_panel!(pos, xdata, ydata, title_str, xl, yl; color=:dodgerblue)
    ρ = round(cor(vec(xdata), vec(ydata)), digits=4)
    ax = Axis(pos; title="$title_str (ρ=$ρ)", xlabel=xl, ylabel=yl, aspect=1)
    scatter!(ax, vec(xdata), vec(ydata); color, markersize=5, alpha=0.6)
    lo, hi = extrema(vcat(vec(xdata), vec(ydata)))
    lines!(ax, [lo, hi], [lo, hi]; color=:red, linestyle=:dash)
    return ρ
end

function heat_panel!(pos, M, title_str)
    ax = Axis(pos; title=title_str, yreversed=true, aspect=DataAspect())
    hm = heatmap!(ax, M')
    Colorbar(pos[1, 2], hm)
end

function free_energy_panel!(pos, F, title_str)
    μ = vec(mean(F; dims=1))
    σ = vec(std(F; dims=1))
    ax = Axis(pos; title=title_str, xlabel="sampling step", ylabel="free energy", aspect=1)
    band!(ax, 1:length(μ), μ .- σ / 2, μ .+ σ / 2; color=(:dodgerblue, 0.3))
    lines!(ax, 1:length(μ), μ; color=:dodgerblue)
end

# =============================================================================
# PER-ARCHITECTURE ANALYSIS
# =============================================================================
summaries = Dict{String,Any}()

for (arch, label) in ARCHS
    println("\n" * "="^60)
    println(label)
    println("="^60)

    pA, pB, pP = hdf5path(arch, "rbm_A", private_suffix()), hdf5path(arch, "rbm_B", private_suffix()), hdf5path(arch, "rbm_paired", paired_suffix())
    if !(isfile(pA) && isfile(pB) && isfile(pP))
        println("  Missing model file(s) for this config — skipping $label.")
        println("    expected: $pA")
        println("              $pB")
        println("              $pP")
        continue
    end

    rbm_A      = gpu(load_rbm(pA))
    rbm_B      = gpu(load_rbm(pB))
    rbm_paired = gpu(load_rbm(pP))

    XA   = gpu(arch == "" ? onehot_potts(XA_states, q) : Float32.(XA_states))
    XB   = gpu(arch == "" ? onehot_potts(XB_states, q) : Float32.(XB_states))
    X_AB = arch == "" ? cat(XA, XB; dims=2) : vcat(XA, XB)

    n_flat_A    = arch == "" ? q * N_VIS : N_VIS
    vis_A_range = 1:n_flat_A
    vis_B_range = (n_flat_A + 1):(2 * n_flat_A)

    n_hid_A       = size(rbm_A.w)[end]
    n_hid_B       = size(rbm_B.w)[end]
    hid_A_range   = 1:n_hid_A
    hid_B_range   = (n_hid_A + 1):(n_hid_A + n_hid_B)
    hid_add_range = (n_hid_A + n_hid_B + 1):(n_hid_A + n_hid_B + H_ADD)

    println("Sampling RBM A ($N_SAMPLES samples, $(N_GIBBS_STEPS*GIBBS_STRIDE) Gibbs sweeps)...")
    fantasy_x_A, F_A = gibbs_sample(rbm_A, N_SAMPLES, N_GIBBS_STEPS, GIBBS_STRIDE)
    println("Sampling RBM B...")
    fantasy_x_B, F_B = gibbs_sample(rbm_B, N_SAMPLES, N_GIBBS_STEPS, GIBBS_STRIDE)
    println("Sampling Paired RBM...")
    fantasy_x_paired, F_P = gibbs_sample(rbm_paired, N_SAMPLES, N_GIBBS_STEPS, GIBBS_STRIDE)
    fantasy_x_paired_A = vis_slice(fantasy_x_paired, 1:N_VIS)
    fantasy_x_paired_B = vis_slice(fantasy_x_paired, (N_VIS + 1):(2 * N_VIS))

    # ---- Free energy (burn-in sanity check) ----
    fig = Figure(size=(3 * CELL, CELL))
    free_energy_panel!(fig[1, 1], F_A, "Free energy — RBM A")
    free_energy_panel!(fig[1, 2], F_B, "Free energy — RBM B")
    free_energy_panel!(fig[1, 3], F_P, "Free energy — Paired RBM")
    save(figpath(arch, "free_energy"), fig)

    # ---- Individual RBM validation (one-hot-level moments — matches what the model was trained on) ----
    h_data_A  = Array(Float64.(sample_h_from_v(rbm_A, XA)))
    h_model_A = Array(Float64.(sample_h_from_v(rbm_A, fantasy_x_A)))
    h_data_B  = Array(Float64.(sample_h_from_v(rbm_B, XB)))
    h_model_B = Array(Float64.(sample_h_from_v(rbm_B, fantasy_x_B)))
    XA_c, XB_c = Array(XA), Array(XB)
    fantasy_x_A_c, fantasy_x_B_c = Array(fantasy_x_A), Array(fantasy_x_B)

    fig = Figure(size=(2 * CELL, 3 * CELL))
    scatter_panel!(fig[1, 1], vec(mean(h_data_A; dims=2)), vec(mean(h_model_A; dims=2)), "⟨h⟩ — RBM A", "data", "model"; color=:dodgerblue)
    scatter_panel!(fig[1, 2], vec(mean(h_data_B; dims=2)), vec(mean(h_model_B; dims=2)), "⟨h⟩ — RBM B", "data", "model"; color=:orange)
    scatter_panel!(fig[2, 1], visible_means(XA_c), visible_means(fantasy_x_A_c), "⟨v⟩ — RBM A", "data", "model"; color=:dodgerblue)
    scatter_panel!(fig[2, 2], visible_means(XB_c), visible_means(fantasy_x_B_c), "⟨v⟩ — RBM B", "data", "model"; color=:orange)
    scatter_panel!(fig[3, 1], vec(hv_moments(h_data_A, XA_c)), vec(hv_moments(h_model_A, fantasy_x_A_c)), "⟨hv⟩ — RBM A", "data", "model"; color=:dodgerblue)
    scatter_panel!(fig[3, 2], vec(hv_moments(h_data_B, XB_c)), vec(hv_moments(h_model_B, fantasy_x_B_c)), "⟨hv⟩ — RBM B", "data", "model"; color=:orange)
    save(figpath(arch, "individual_validation"), fig)

    # ---- Individual RBM pairwise correlations (site-level, decoded — comparable across architectures) ----
    CA_data  = cor(decode_state(XA_c)')
    CB_data  = cor(decode_state(XB_c)')
    CA_model = cor(decode_state(fantasy_x_A_c)')
    CB_model = cor(decode_state(fantasy_x_B_c)')

    fig = Figure(size=(2 * CELL_HEAT, 2 * CELL))
    heat_panel!(fig[1, 1], CA_data,  "Corr data — A")
    heat_panel!(fig[1, 2], CA_model, "Corr model — A")
    heat_panel!(fig[2, 1], CB_data,  "Corr data — B")
    heat_panel!(fig[2, 2], CB_model, "Corr model — B")
    save(figpath(arch, "individual_correlations"), fig)

    # ---- Same comparison as a scatter of correlation *values* (data vs ----
    # ---- model, point per pair), the actual point cloud behind vv_check's r.
    fig = Figure(size=(2 * CELL, CELL))
    ρ_vv_A = scatter_panel!(fig[1, 1], offdiag_vec(CA_data), offdiag_vec(CA_model), "⟨v_i v_j⟩ — family A", "data", "model"; color=:dodgerblue)
    ρ_vv_B = scatter_panel!(fig[1, 2], offdiag_vec(CB_data), offdiag_vec(CB_model), "⟨v_i v_j⟩ — family B", "data", "model"; color=:orange)
    save(figpath(arch, "individual_vv_scatter"), fig)

    # ---- Paired RBM — full model moments ----
    h_data_P  = Array(Float64.(sample_h_from_v(rbm_paired, X_AB)))
    h_model_P = Array(Float64.(sample_h_from_v(rbm_paired, fantasy_x_paired)))
    X_AB_c, fantasy_P_c = Array(X_AB), Array(fantasy_x_paired)
    X_AB_flat, fantasy_P_flat = flatten(X_AB_c), flatten(fantasy_P_c)
    n_data, n_samp = size(X_AB_flat, 2), size(fantasy_P_flat, 2)

    fig = Figure(size=(3 * CELL, CELL))
    scatter_panel!(fig[1, 1], vec(mean(h_data_P; dims=2)), vec(mean(h_model_P; dims=2)), "⟨h⟩ — Paired RBM", "data", "model"; color=:purple)
    ρ_v_paired = scatter_panel!(fig[1, 2], visible_means(X_AB_c), visible_means(fantasy_P_c), "⟨v⟩ — Paired RBM", "data", "model"; color=:purple)
    hv_data_P  = h_data_P  * X_AB_flat' / n_data
    hv_model_P = h_model_P * fantasy_P_flat' / n_samp
    scatter_panel!(fig[1, 3], vec(hv_data_P), vec(hv_model_P), "⟨hv⟩ — Paired RBM (all)", "data", "model"; color=:purple)
    save(figpath(arch, "paired_full_moments"), fig)

    # ---- Paired RBM — block-wise ⟨hv⟩ moments: the direct test of what the ----
    # ---- added hidden units learned, versus what stayed correctly frozen. ----
    hv_data_AA  = h_data_P[hid_A_range, :]  * X_AB_flat[vis_A_range, :]' / n_data
    hv_model_AA = h_model_P[hid_A_range, :] * fantasy_P_flat[vis_A_range, :]' / n_samp
    hv_data_BB  = h_data_P[hid_B_range, :]  * X_AB_flat[vis_B_range, :]' / n_data
    hv_model_BB = h_model_P[hid_B_range, :] * fantasy_P_flat[vis_B_range, :]' / n_samp
    hv_data_addA  = h_data_P[hid_add_range, :]  * X_AB_flat[vis_A_range, :]' / n_data
    hv_model_addA = h_model_P[hid_add_range, :] * fantasy_P_flat[vis_A_range, :]' / n_samp
    hv_data_addB  = h_data_P[hid_add_range, :]  * X_AB_flat[vis_B_range, :]' / n_data
    hv_model_addB = h_model_P[hid_add_range, :] * fantasy_P_flat[vis_B_range, :]' / n_samp
    # added units vs the *whole* visible layer at once (both families together)
    hv_data_addfull  = h_data_P[hid_add_range, :]  * X_AB_flat' / n_data
    hv_model_addfull = h_model_P[hid_add_range, :] * fantasy_P_flat' / n_samp

    fig = Figure(size=(3 * CELL, 2 * CELL))
    ρ_AA      = scatter_panel!(fig[1, 1], hv_data_AA,      hv_model_AA,      "⟨h_A v_A⟩",      "data", "model"; color=:dodgerblue)
    ρ_BB      = scatter_panel!(fig[1, 2], hv_data_BB,      hv_model_BB,      "⟨h_B v_B⟩",      "data", "model"; color=:orange)
    ρ_addfull = scatter_panel!(fig[1, 3], hv_data_addfull, hv_model_addfull, "⟨h_add v_full⟩", "data", "model"; color=:seagreen)
    ρ_addA    = scatter_panel!(fig[2, 1], hv_data_addA,    hv_model_addA,    "⟨h_add v_A⟩",    "data", "model"; color=:seagreen)
    ρ_addB    = scatter_panel!(fig[2, 2], hv_data_addB,    hv_model_addB,    "⟨h_add v_B⟩",    "data", "model"; color=:seagreen)
    save(figpath(arch, "paired_block_hv"), fig)

    # ---- Cross-correlation vs ground truth K (the definitive test) ----
    OA_data, OB_data = decode_state(XA_c)', decode_state(XB_c)'
    OA_paired, OB_paired = decode_state(fantasy_x_paired_A)', decode_state(fantasy_x_paired_B)'
    OA_indep, OB_indep = decode_state(fantasy_x_A_c)', decode_state(fantasy_x_B_c)'  # independent RBMs = zero-coupling baseline

    C_AB_data   = empirical_cross_corr(OA_data,   OB_data)
    C_AB_paired = empirical_cross_corr(OA_paired, OB_paired)
    C_AB_indep  = empirical_cross_corr(OA_indep,  OB_indep)

    fig = Figure(size=(2 * CELL_HEAT, 2 * CELL))
    heat_panel!(fig[1, 1], K_true,      "True K (A-B)")
    heat_panel!(fig[1, 2], C_AB_data,   "Data cross-corr (A-B)")
    heat_panel!(fig[2, 1], C_AB_paired, "Paired RBM (A-B)")
    heat_panel!(fig[2, 2], C_AB_indep,  "Independent RBMs (A-B)")
    save(figpath(arch, "cross_correlation_vs_truth"), fig)

    threshold  = 1e-6
    true_pairs = findall(x -> abs(x) > threshold, K_true)
    Kn         = length(true_pairs)
    top_true   = sortperm(vec(abs.(K_true)),      rev=true)[1:Kn]
    top_paired = sortperm(vec(abs.(C_AB_paired)), rev=true)[1:Kn]
    top_data   = sortperm(vec(abs.(C_AB_data)),   rev=true)[1:Kn]
    top_indep  = sortperm(vec(abs.(C_AB_indep)),  rev=true)[1:Kn]

    cor_paired_vs_K = cor(vec(C_AB_paired), vec(K_true))
    cor_data_vs_K   = cor(vec(C_AB_data),   vec(K_true))
    cor_indep_vs_K  = cor(vec(C_AB_indep),  vec(K_true))
    overlap_paired  = length(intersect(top_true, top_paired)) / Kn
    overlap_data    = length(intersect(top_true, top_data))   / Kn
    overlap_indep   = length(intersect(top_true, top_indep))  / Kn

    # ---- Family preservation: does pairing corrupt each family's own stats? ----
    h_data_A_pres  = Array(Float64.(sample_h_from_v(rbm_A, XA)))
    h_model_A_pres = Array(Float64.(sample_h_from_v(rbm_A, fantasy_x_paired_A)))
    h_data_B_pres  = Array(Float64.(sample_h_from_v(rbm_B, XB)))
    h_model_B_pres = Array(Float64.(sample_h_from_v(rbm_B, fantasy_x_paired_B)))
    fantasy_x_paired_A_c, fantasy_x_paired_B_c = Array(fantasy_x_paired_A), Array(fantasy_x_paired_B)

    fig = Figure(size=(3 * CELL, 2 * CELL))
    scatter_panel!(fig[1, 1], vec(mean(h_data_A_pres; dims=2)), vec(mean(h_model_A_pres; dims=2)), "⟨h⟩ family A", "data", "model"; color=:dodgerblue)
    scatter_panel!(fig[1, 2], visible_means(XA_c), visible_means(fantasy_x_paired_A_c), "⟨v⟩ family A", "data", "model"; color=:dodgerblue)
    scatter_panel!(fig[1, 3], vec(hv_moments(h_data_A_pres, XA_c)), vec(hv_moments(h_model_A_pres, fantasy_x_paired_A_c)), "⟨hv⟩ family A", "data", "model"; color=:dodgerblue)
    scatter_panel!(fig[2, 1], vec(mean(h_data_B_pres; dims=2)), vec(mean(h_model_B_pres; dims=2)), "⟨h⟩ family B", "data", "model"; color=:orange)
    scatter_panel!(fig[2, 2], visible_means(XB_c), visible_means(fantasy_x_paired_B_c), "⟨v⟩ family B", "data", "model"; color=:orange)
    scatter_panel!(fig[2, 3], vec(hv_moments(h_data_B_pres, XB_c)), vec(hv_moments(h_model_B_pres, fantasy_x_paired_B_c)), "⟨hv⟩ family B", "data", "model"; color=:orange)
    save(figpath(arch, "family_preservation"), fig)

    CA_paired_model = cor(decode_state(fantasy_x_paired_A_c)')
    CB_paired_model = cor(decode_state(fantasy_x_paired_B_c)')

    fig = Figure(size=(2 * CELL_HEAT, 2 * CELL))
    heat_panel!(fig[1, 1], CA_data,          "Corr data — A")
    heat_panel!(fig[1, 2], CA_paired_model,  "Corr paired — A")
    heat_panel!(fig[2, 1], CB_data,          "Corr data — B")
    heat_panel!(fig[2, 2], CB_paired_model,  "Corr paired — B")
    save(figpath(arch, "family_correlation_preservation"), fig)

    summaries[label] = (;
        ρ_AA, ρ_BB, ρ_addA, ρ_addB, ρ_addfull, ρ_vv_A, ρ_vv_B, ρ_v_paired,
        Kn, overlap_paired, overlap_data, overlap_indep,
        cor_paired_vs_K, cor_data_vs_K, cor_indep_vs_K,
        preservation_A_data = offdiag_mean_abs(CA_data),
        preservation_A_paired = offdiag_mean_abs(CA_paired_model),
        preservation_B_data = offdiag_mean_abs(CB_data),
        preservation_B_paired = offdiag_mean_abs(CB_paired_model),
    )

    println("\nBlock-wise ⟨hv⟩ correlations (data vs model, from real equilibrium samples):")
    println("  ⟨h_A  v_A⟩    ρ = $ρ_AA   (frozen block — should be ≈1)")
    println("  ⟨h_B  v_B⟩    ρ = $ρ_BB   (frozen block — should be ≈1)")
    println("  ⟨h_add v_A⟩   ρ = $ρ_addA (learned cross-family signal)")
    println("  ⟨h_add v_B⟩   ρ = $ρ_addB (learned cross-family signal)")
    println("  ⟨h_add v_full⟩ ρ = $ρ_addfull (learned cross-family signal, whole visible layer at once)")
    println("  ⟨v⟩ paired (single-site marginal) ρ = $ρ_v_paired (collateral shift from added units — see paired_full_moments)")
    println("\nIndividual RBM visible-visible correlation, data vs model (real equilibrium samples):")
    println("  family A  ρ = $ρ_vv_A")
    println("  family B  ρ = $ρ_vv_B")
    println("\nCross-correlation vs ground truth K:")
    println("  top-$Kn overlap — paired RBM vs true: $(round(overlap_paired; digits=3))")
    println("  top-$Kn overlap — data vs true:       $(round(overlap_data; digits=3))")
    println("  top-$Kn overlap — independent vs true: $(round(overlap_indep; digits=3))")
    println("  corr(paired, K_true) = $(round(cor_paired_vs_K; digits=3))  corr(data, K_true) = $(round(cor_data_vs_K; digits=3))  corr(indep, K_true) = $(round(cor_indep_vs_K; digits=3))")
end

# =============================================================================
# WRITTEN REPORT
# =============================================================================
io = IOBuffer()
rp(x...) = println(io, x...)
rp("# Sampled-model validation — N_HIDDEN=$(N_HIDDEN), H_ADD=$(H_ADD), N_ITERS=$(N_ITERS), PAIRED_ITERS=$(PAIRED_ITERS)")
rp()
rp("Real Gibbs samples ($N_SAMPLES samples, $(N_GIBBS_STEPS*GIBBS_STRIDE) sweeps burn-in each) from the actual saved models —")
rp("not the short training-time chains `analyze_results.jl` reports on. Figures in `$(REPORT_DIR)/`.")
rp()
for (label, s) in summaries
    rp("## $label")
    rp()
    rp("**Block-wise ⟨hv⟩ correlation (data vs model, real samples):**")
    rp("- frozen blocks: ⟨h_A v_A⟩ ρ=$(round(s.ρ_AA;digits=3)), ⟨h_B v_B⟩ ρ=$(round(s.ρ_BB;digits=3)) — should be ≈1")
    rp("- **added units — cross-family signal**: ⟨h_add v_A⟩ ρ=$(round(s.ρ_addA;digits=3)), ⟨h_add v_B⟩ ρ=$(round(s.ρ_addB;digits=3)), ⟨h_add v_full⟩ ρ=$(round(s.ρ_addfull;digits=3))")
    rp("- **⟨v⟩ paired (single-site marginal, collateral shift)**: ρ=$(round(s.ρ_v_paired;digits=3))")
    rp()
    rp("**Individual RBM visible-visible correlation, data vs model (real samples, see `individual_vv_scatter`):**")
    rp("- family A ρ=$(round(s.ρ_vv_A;digits=3)), family B ρ=$(round(s.ρ_vv_B;digits=3))")
    rp()
    rp("**Cross-family correlation vs ground-truth K (the definitive test):**")
    rp("- top-$(s.Kn) strongest-pair overlap: paired RBM=$(round(s.overlap_paired;digits=3)), data=$(round(s.overlap_data;digits=3)), independent-RBM baseline=$(round(s.overlap_indep;digits=3))")
    rp("- correlation of full cross-corr matrix against K_true: paired=$(round(s.cor_paired_vs_K;digits=3)), data=$(round(s.cor_data_vs_K;digits=3)), independent=$(round(s.cor_indep_vs_K;digits=3))")
    gap = s.overlap_paired - s.overlap_indep
    rp("  → paired model beats the zero-coupling (independent-RBM) baseline by $(round(gap;digits=3)) in top-K overlap" *
       (gap > 0.1 ? " — real cross-family structure recovered." : " — little to no improvement over having learned nothing about family coupling."))
    rp()
    rp("**Family preservation (off-diagonal |corr|, lower is tighter to data):**")
    rp("- A: data=$(round(s.preservation_A_data;digits=4)), within paired model=$(round(s.preservation_A_paired;digits=4))")
    rp("- B: data=$(round(s.preservation_B_data;digits=4)), within paired model=$(round(s.preservation_B_paired;digits=4))")
    rp()
end

if length(summaries) == 2
    labels = collect(keys(summaries))
    rp("## Potts vs Binary — head-to-head (real samples)")
    rp()
    rp("| metric | $(labels[1]) | $(labels[2]) |")
    rp("|---|---|---|")
    rp("| ⟨h_add v_A⟩ ρ | $(round(summaries[labels[1]].ρ_addA;digits=3)) | $(round(summaries[labels[2]].ρ_addA;digits=3)) |")
    rp("| ⟨h_add v_B⟩ ρ | $(round(summaries[labels[1]].ρ_addB;digits=3)) | $(round(summaries[labels[2]].ρ_addB;digits=3)) |")
    rp("| ⟨h_add v_full⟩ ρ | $(round(summaries[labels[1]].ρ_addfull;digits=3)) | $(round(summaries[labels[2]].ρ_addfull;digits=3)) |")
    rp("| top-K overlap vs true K (paired) | $(round(summaries[labels[1]].overlap_paired;digits=3)) | $(round(summaries[labels[2]].overlap_paired;digits=3)) |")
    rp("| corr(paired cross-corr, K_true) | $(round(summaries[labels[1]].cor_paired_vs_K;digits=3)) | $(round(summaries[labels[2]].cor_paired_vs_K;digits=3)) |")
end

report = String(take!(io))
open(joinpath(REPORT_DIR, "report.md"), "w") do f
    write(f, report)
end
println("\n" * report)
println("Report + figures written to $(REPORT_DIR)/")
