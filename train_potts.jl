using CUDA
using HDF5
using RestrictedBoltzmannMachines: Potts,PottsGumbel, RBM, initialize!, standardize, unstandardize
# using RestrictedBoltzmannMachines: nsReLU
using RestrictedBoltzmannMachines: xReLU
using RestrictedBoltzmannMachines: log_pseudolikelihood, pcd!, save_rbm, load_rbm
using RestrictedBoltzmannMachines: sample_h_from_v, sample_v_from_v, mean_h_from_v, gpu, cpu
using RestrictedBoltzmannMachines: ∂free_energy
using Optimisers: Adam, ClipNorm, OptimiserChain
using Statistics: mean
using Random
using TwoFamilyIsing
using OneHotArrays

# =============================================================================
# CONFIG
# =============================================================================
# ARGS: N_ITERS PAIRED_ITERS [H_ADD=30] [N_HIDDEN=30] [REG=0] [DATASET_TAG=20links_09strength] [SEED=42] [LR=1e-3]
const N_ITERS      = parse(Int, ARGS[1])
const PAIRED_ITERS = parse(Int, ARGS[2])
const H_ADD        = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 30
const N_HIDDEN     = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 30
const REG          = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 0
const DATASET_TAG  = length(ARGS) >= 6 ? ARGS[6] : "20links_09strength"
const SEED         = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : 42
const LR           = length(ARGS) >= 8 ? parse(Float64, ARGS[8]) : 1e-3
const DATA_PATH    = "./data/dataset_$(DATASET_TAG).bin"
const OUTPUT_DIR   = "./results"
isdir(OUTPUT_DIR) || mkdir(OUTPUT_DIR)

Random.seed!(SEED)

const BATCH_SIZE  = 512
const CD_STEPS    = 50
const LOG_EVERY   = 100
const CLIP_NORM   = 1.0

# save_rbm errors instead of overwriting if the file already exists (e.g. a
# rerun of an already-completed seed/H_ADD combination) — remove any stale
# file first so a rerun can always replace it.
function save_rbm_force(path, rbm)
    isfile(path) && rm(path)
    save_rbm(path, rbm)
end

regstr(r) = isinteger(r) ? string(Int(r)) : string(r)  # REG=0 (not 0.0) matches pre-existing filenames
# Hidden-unit type is tagged explicitly so this run's files land next to,
# rather than overwrite, any pre-existing nsReLU results at the same
# N_HIDDEN/H_ADD/N_ITERS/REG/DATASET_TAG combination.
const HID_TAG = "xReLU"
# Seed only enters the filename when non-default, so ordinary seed=42 runs
# keep their existing paths — a second seed is tagged explicitly to sit
# alongside, not overwrite, the original run.
seedsuffix() = SEED == 42 ? "" : "_SEED=$(SEED)"
# Same convention for the learning rate: untagged at the original default
# (1e-3) so existing results keep their paths; tagged explicitly otherwise.
lrsuffix() = LR == 1e-3 ? "" : "_LR=$(LR)"
# Private RBM training (rbm_A, rbm_B) does not depend on H_ADD at all —
# H_ADD only ever affects the paired model's construction. Giving the
# private RBMs their own H_ADD-free suffix means the same trained rbm_A/
# rbm_B can be reused across an entire sweep over H_ADD at fixed seed,
# instead of uselessly retraining two full private RBMs (the bulk of the
# runtime) once per H_ADD value.
private_suffix() = "N_HIDDEN=$(N_HIDDEN)_K=$(CD_STEPS)_N_ITERS=$(N_ITERS)_REG=$(regstr(REG))_DATA=$(DATASET_TAG)_HID=$(HID_TAG)$(seedsuffix())$(lrsuffix())"
single_suffix() = "N_HIDDEN=$(N_HIDDEN)_H_ADD=$(H_ADD)_K=$(CD_STEPS)_N_ITERS=$(N_ITERS)_REG=$(regstr(REG))_DATA=$(DATASET_TAG)_HID=$(HID_TAG)$(seedsuffix())$(lrsuffix())"
paired_suffix()  = single_suffix() * "_PAIRED_ITERS=$(PAIRED_ITERS)"

path_rbm_A      = joinpath(OUTPUT_DIR, "rbm_A_$(private_suffix()).hdf5")
path_rbm_B      = joinpath(OUTPUT_DIR, "rbm_B_$(private_suffix()).hdf5")
path_rbm_paired = joinpath(OUTPUT_DIR, "rbm_paired_$(paired_suffix()).hdf5")
path_lpl_A      = joinpath(OUTPUT_DIR, "lpl_A_$(private_suffix()).txt")
path_lpl_B      = joinpath(OUTPUT_DIR, "lpl_B_$(private_suffix()).txt")
path_vh_A       = joinpath(OUTPUT_DIR, "vh_check_A_$(private_suffix()).txt")
path_vh_B       = joinpath(OUTPUT_DIR, "vh_check_B_$(private_suffix()).txt")
path_v_A        = joinpath(OUTPUT_DIR, "v_check_A_$(private_suffix()).txt")
path_v_B        = joinpath(OUTPUT_DIR, "v_check_B_$(private_suffix()).txt")
path_wn_A       = joinpath(OUTPUT_DIR, "wn_check_A_$(private_suffix()).txt")
path_wn_B       = joinpath(OUTPUT_DIR, "wn_check_B_$(private_suffix()).txt")
path_lpl_paired = joinpath(OUTPUT_DIR, "lpl_paired_$(paired_suffix()).txt")
path_freeze_paired = joinpath(OUTPUT_DIR, "freeze_check_paired_$(paired_suffix()).txt")
path_vh_paired  = joinpath(OUTPUT_DIR, "vh_check_paired_$(paired_suffix()).txt")
path_ab_paired  = joinpath(OUTPUT_DIR, "ab_check_paired_$(paired_suffix()).txt")
path_firing_paired = joinpath(OUTPUT_DIR, "firing_check_paired_$(paired_suffix()).txt")
path_v_paired   = joinpath(OUTPUT_DIR, "v_check_paired_$(paired_suffix()).txt")
path_wn_paired  = joinpath(OUTPUT_DIR, "wn_check_paired_$(paired_suffix()).txt")
path_lplval_A       = joinpath(OUTPUT_DIR, "lplval_A_$(private_suffix()).txt")
path_lplval_B       = joinpath(OUTPUT_DIR, "lplval_B_$(private_suffix()).txt")
path_lplval_paired  = joinpath(OUTPUT_DIR, "lplval_paired_$(paired_suffix()).txt")
path_gn_A       = joinpath(OUTPUT_DIR, "gn_check_A_$(private_suffix()).txt")
path_gn_B       = joinpath(OUTPUT_DIR, "gn_check_B_$(private_suffix()).txt")
path_gn_paired  = joinpath(OUTPUT_DIR, "gn_check_paired_$(paired_suffix()).txt")

# =============================================================================
# DATA
# =============================================================================
data     = load_dataset(DATA_PATH)
XA_spins = data.XA_train'
XB_spins = data.XB_train'

XA_states = @. Int((XA_spins + 1) / 2)
XB_states = @. Int((XB_spins + 1) / 2)

q       = maximum(XA_states) + 1
N_VIS   = size(XA_states, 1)

function onehot_potts(states, q)
    n_sites, n_samples = size(states)
    out = falses(q, n_sites, n_samples)
    for s in 1:n_sites
        out[:, s, :] = onehotbatch(states[s, :], 0:(q-1))
    end
    return out
end

XA   = onehot_potts(XA_states, q)
XB   = onehot_potts(XB_states, q)
X_AB = cat(XA, XB; dims=2)

println("XA: $(size(XA))  XB: $(size(XB))  X_AB: $(size(X_AB))")

# Held-out validation split (data.XA_val/XB_val), never trained on. Used only
# to check that the pseudolikelihood improvement seen on `vd` (a training
# minibatch) actually generalizes, rather than just reflecting memorization —
# see the HELD-OUT PSEUDOLIKELIHOOD CHECKPOINT section below.
XA_val_states = @. Int((data.XA_val' + 1) / 2)
XB_val_states = @. Int((data.XB_val' + 1) / 2)
XA_val   = onehot_potts(XA_val_states, q)
XB_val   = onehot_potts(XB_val_states, q)
X_AB_val = cat(XA_val, XB_val; dims=2)

println("XA_val: $(size(XA_val))  XB_val: $(size(XB_val))  X_AB_val: $(size(X_AB_val))")

# =============================================================================
# PARAMETER PROJECTION
# =============================================================================
function frozen_block_ranges(rbm_A, rbm_B)
    n_vis_A, n_hid_A = size(rbm_A.w, 2), size(rbm_A.w, 3)
    n_vis_B, n_hid_B = size(rbm_B.w, 2), size(rbm_B.w, 3)
    n_vis_total = n_vis_A + n_vis_B

    vis_A = 1:n_vis_A
    vis_B = (n_vis_A + 1):n_vis_total
    hid_A = 1:n_hid_A
    hid_B = (n_hid_A + 1):(n_hid_A + n_hid_B)
    return (; vis_A, vis_B, hid_A, hid_B)
end

function project_to_frozen_par!(rbm_paired, rbm_A, rbm_B, h_add)
    (; vis_A, vis_B, hid_A, hid_B) = frozen_block_ranges(rbm_A, rbm_B)

    rbm_paired.visible.par[:, :, vis_A] .= rbm_A.visible.par
    rbm_paired.visible.par[:, :, vis_B] .= rbm_B.visible.par
    rbm_paired.hidden.par[:, hid_A]     .= rbm_A.hidden.par
    rbm_paired.hidden.par[:, hid_B]     .= rbm_B.hidden.par

    rbm_paired.w[:, vis_A, hid_A] .= rbm_A.w
    rbm_paired.w[:, vis_A, hid_B] .= 0
    rbm_paired.w[:, vis_B, hid_A] .= 0
    rbm_paired.w[:, vis_B, hid_B] .= rbm_B.w
end

# =============================================================================
# FREEZE CHECKPOINT
# =============================================================================
# Since `pcd!` runs the Adam update on *all* parameters every iteration,
# freezing rbm_A's/rbm_B's blocks inside rbm_paired is implemented by resetting
# them back to the reference values after each step (see `freeze_callback`
# below), rather than by excluding them from the gradient. `freeze_drift`
# measures, for each frozen block, how far its current value sits from what it
# should be. Called right before the reset it shows how much a single Adam
# step tried to move a supposedly-frozen block (should be small but nonzero);
# called right after the reset it must be exactly zero, or the freeze is
# broken (e.g. wrong index ranges, aliasing, an array not actually written).
const FREEZE_TOL = 1f-6

function freeze_drift(rbm_paired, rbm_A, rbm_B)
    (; vis_A, vis_B, hid_A, hid_B) = frozen_block_ranges(rbm_A, rbm_B)
    return (
        vis_A = maximum(abs, rbm_paired.visible.par[:, :, vis_A] .- rbm_A.visible.par),
        vis_B = maximum(abs, rbm_paired.visible.par[:, :, vis_B] .- rbm_B.visible.par),
        hid_A = maximum(abs, rbm_paired.hidden.par[:, hid_A] .- rbm_A.hidden.par),
        hid_B = maximum(abs, rbm_paired.hidden.par[:, hid_B] .- rbm_B.hidden.par),
        w_AA  = maximum(abs, rbm_paired.w[:, vis_A, hid_A] .- rbm_A.w),
        w_BB  = maximum(abs, rbm_paired.w[:, vis_B, hid_B] .- rbm_B.w),
        w_AB  = maximum(abs, rbm_paired.w[:, vis_A, hid_B]),  # cross block, must stay 0
        w_BA  = maximum(abs, rbm_paired.w[:, vis_B, hid_A]),  # cross block, must stay 0
    )
end

function log_freeze_checkpoint(path_freeze, iter, drift_before, drift_after)
    worst_before = maximum(drift_before)
    worst_after  = maximum(drift_after)
    status = worst_after <= FREEZE_TOL ? "OK" : "VIOLATION"
    open(path_freeze, "a") do f
        println(f, "iter=$iter status=$status drift_before_reset=$drift_before drift_after_reset=$drift_after")
    end
    println("iter=$iter  freeze_check=$status  max_drift_before_reset=$worst_before  max_drift_after_reset=$worst_after")
    if worst_after > FREEZE_TOL
        @warn "Frozen parameters did not reset to their reference values" iter drift_after
    end
end

# =============================================================================
# VH CORRELATION CHECKPOINT (are the added hidden units learning?)
# =============================================================================
# For an RBM, the weight gradient is driven by the discrepancy between the
# data-driven and model-driven visible-hidden correlations, <v h>_data and
# <v h>_model (the sufficient statistics of contrastive divergence). If that
# discrepancy is ~0 throughout training for a hidden unit, its incoming
# weights get essentially no learning signal — either because it has already
# converged, or because it is "dead" (e.g. a ReLU stuck at 0 for both data and
# fantasy particles). We track this specifically for the H_ADD extra hidden
# units in rbm_paired, since those are the only ones actually free to learn
# (see FREEZE CHECKPOINT above).
function added_hidden_range(rbm_A, rbm_B, h_add)
    n_hid_A = size(rbm_A.w, 3)
    n_hid_B = size(rbm_B.w, 3)
    return (n_hid_A + n_hid_B + 1):(n_hid_A + n_hid_B + h_add)
end

# <v h> averaged over the batch: v is (q, n_vis, batch), h is (n_hid, batch);
# returns (q, n_vis, n_hid).
function vh_correlation(v, h)
    q, n_vis, batch = size(v)
    vr = reshape(Float32.(v), q * n_vis, batch)
    C  = (vr * h') ./ batch
    return reshape(C, q, n_vis, size(h, 1))
end

# Ordinary least-squares fit of y ~ slope*x + intercept, plus Pearson r.
# A perfectly-trained RBM has <vh>_model tracking <vh>_data along y=x, i.e.
# r → 1, slope → 1, intercept → 0 — not merely a small |x - y|, which can look
# deceptively good if both are just small and noisy.
function linear_alignment(x, y)
    mx, my = mean(x), mean(y)
    cov_xy = mean((x .- mx) .* (y .- my))
    var_x  = mean((x .- mx) .^ 2)
    var_y  = mean((y .- my) .^ 2)
    r         = cov_xy / sqrt(var_x * var_y)
    slope     = cov_xy / var_x
    intercept = my - slope * mx
    return (; r, slope, intercept)
end

function vh_learning_checkpoint(rbm, vd, vm, hid_add)
    hd = mean_h_from_v(rbm, vd)[hid_add, :]
    hm = mean_h_from_v(rbm, vm)[hid_add, :]

    cd = vh_correlation(vd, hd)
    cm = vh_correlation(vm, hm)
    diff  = cd .- cm
    align = linear_alignment(vec(cd), vec(cm))

    return (
        mean_abs_diff = mean(abs, diff),
        max_abs_diff  = maximum(abs, diff),
        r             = align.r,
        slope         = align.slope,
        intercept     = align.intercept,
        mean_h_data   = Array(vec(mean(hd; dims=2))),
        mean_h_model  = Array(vec(mean(hm; dims=2))),
    )
end

function log_vh_checkpoint(path_vh, iter, chk)
    open(path_vh, "a") do f
        println(f, "iter=$iter mean_abs_diff=$(chk.mean_abs_diff) max_abs_diff=$(chk.max_abs_diff) r=$(chk.r) slope=$(chk.slope) intercept=$(chk.intercept) mean_h_data=$(chk.mean_h_data) mean_h_model=$(chk.mean_h_model)")
    end
    println("iter=$iter  vh_check  mean|<vh>_data-<vh>_model|=$(chk.mean_abs_diff)  r=$(chk.r)  slope=$(chk.slope)  intercept=$(chk.intercept)")
end

# =============================================================================
# A-B CORRELATION CHECKPOINT (is joint training capturing cross-family structure?)
# =============================================================================
# The whole point of rbm_paired's extra hidden units is to let the model
# represent correlations *between* family A and family B — the frozen private
# blocks can't touch these by construction (the A-B weight cross-blocks are
# pinned at 0, see FREEZE CHECKPOINT above). So the metric that actually
# answers "did joint training help" is the connected two-point correlation
# between visible sites of A and visible sites of B, compared between data and
# model samples — not just something about the added hidden units themselves.
#
# For Potts/one-hot visible units, the connected correlation between site i
# (color a) and site j (color b) is
#   C[a,i,b,j] = <v_i=a v_j=b> - <v_i=a><v_j=b>
# computed once under the data distribution (from vd) and once under the
# model (from the persistent fantasy chains vm), restricted to i ∈ family A,
# j ∈ family B. In a well-trained joint model these two tensors should align
# along y=x (r → 1, slope → 1, intercept → 0), exactly like the vh check above.
function vv_connected_correlation(v_i, v_j)
    q, n_i, batch = size(v_i)
    n_j = size(v_j, 2)
    vi = reshape(Float32.(v_i), q * n_i, batch)
    vj = reshape(Float32.(v_j), q * n_j, batch)
    mean_i = vec(mean(vi; dims=2))
    mean_j = vec(mean(vj; dims=2))
    cross  = (vi * vj') ./ batch
    connected = cross .- mean_i * mean_j'
    return reshape(connected, q, n_i, q, n_j)
end

function ab_correlation_checkpoint(vd, vm, vis_A, vis_B)
    c_data  = vv_connected_correlation(vd[:, vis_A, :], vd[:, vis_B, :])
    c_model = vv_connected_correlation(vm[:, vis_A, :], vm[:, vis_B, :])
    diff  = c_data .- c_model
    align = linear_alignment(vec(c_data), vec(c_model))

    return (
        mean_abs_diff = mean(abs, diff),
        max_abs_diff  = maximum(abs, diff),
        r             = align.r,
        slope         = align.slope,
        intercept     = align.intercept,
    )
end

function log_ab_checkpoint(path_ab, iter, chk)
    open(path_ab, "a") do f
        println(f, "iter=$iter mean_abs_diff=$(chk.mean_abs_diff) max_abs_diff=$(chk.max_abs_diff) r=$(chk.r) slope=$(chk.slope) intercept=$(chk.intercept)")
    end
    println("iter=$iter  ab_check  mean|C_data-C_model|=$(chk.mean_abs_diff)  r=$(chk.r)  slope=$(chk.slope)  intercept=$(chk.intercept)")
end

# =============================================================================
# V CHECKPOINT (does the model reproduce the data's *single-site* marginals —
# the simplest possible target, and the one that turned out to look worst)
# =============================================================================
# <v> per (site, color) is just the marginal probability of each state — the
# weakest possible summary statistic of the visible layer. This replaces the
# previous pairwise vv_check here: evaluated on long, properly-equilibrated
# Gibbs samples (pair_results.jl), the pairwise <v_i v_j> check turned out to
# be fine (r~0.94) — its low training-time number was a short-chain mixing
# artifact — but the *same* equilibrated samples showed <v> itself scattering
# badly against y=x (r as low as 0.12). That is a different failure mode: the
# true site-to-site variation in <v> is tiny (weak fields, range ±0.05)
# relative to the sampling noise of a finite batch, so this check is far more
# sensitive to under-sampling than the pairwise one was — worth tracking
# directly during training rather than assuming it behaves like vv did.
function v_self_checkpoint(vd, vm)
    q, n_vis, batch = size(vd)
    xd = reshape(Float32.(vd), q * n_vis, batch)
    xm = reshape(Float32.(vm), q * n_vis, batch)
    x = vec(mean(xd; dims=2))
    y = vec(mean(xm; dims=2))
    diff  = x .- y
    align = linear_alignment(x, y)

    return (
        mean_abs_diff = mean(abs, diff),
        max_abs_diff  = maximum(abs, diff),
        r             = align.r,
        slope         = align.slope,
        intercept     = align.intercept,
    )
end

function log_v_checkpoint(path_v, iter, chk)
    open(path_v, "a") do f
        println(f, "iter=$iter mean_abs_diff=$(chk.mean_abs_diff) max_abs_diff=$(chk.max_abs_diff) r=$(chk.r) slope=$(chk.slope) intercept=$(chk.intercept)")
    end
    println("iter=$iter  v_check  mean|<v>_data-<v>_model|=$(chk.mean_abs_diff)  r=$(chk.r)  slope=$(chk.slope)  intercept=$(chk.intercept)")
end

# =============================================================================
# FIRING-RATE CHECKPOINT (per added-unit activity, distinct from alignment)
# =============================================================================
# xReLU units are a dReLU mixture: each draw lands either on the positive
# branch (h > 0) or the negative branch (h < 0), with a mixture weight that
# depends on the input. `r`/`slope` in the vh check can look reasonable for a
# unit that is nevertheless collapsed — e.g. one that fires on almost every
# sample regardless of v, or almost never does — because those alignment
# stats are computed from *mean* activations, which don't reveal how
# concentrated the underlying samples are. Firing rate = fraction of samples
# with h_j > 0, computed separately from actual samples (not mean-field
# values) of data (vd) and model (vm). A unit stuck near 0 or 1 for both,
# and staying that way across checkpoints, isn't discriminating between
# inputs — regardless of what its mean-based correlation stats say.
function firing_rate_checkpoint(rbm, vd, vm, hid_add)
    hd = sample_h_from_v(rbm, vd)[hid_add, :]
    hm = sample_h_from_v(rbm, vm)[hid_add, :]
    return (
        data_rate  = Array(vec(mean(hd .> 0; dims=2))),
        model_rate = Array(vec(mean(hm .> 0; dims=2))),
    )
end

function log_firing_checkpoint(path_firing, iter, chk)
    open(path_firing, "a") do f
        println(f, "iter=$iter data_rate=$(chk.data_rate) model_rate=$(chk.model_rate)")
    end
    println("iter=$iter  firing_check  data_rate=$(chk.data_rate)  model_rate=$(chk.model_rate)")
end

# =============================================================================
# WEIGHT-NORM CHECKPOINT (are the tracked units acquiring structure at all?)
# =============================================================================
# The vh/vv/firing checks all measure *what a unit is doing with its current
# weights*. This instead tracks the weights themselves: the per-unit Frobenius
# norm of the incoming visible weight column, ‖w[:, :, j]‖, for each unit in
# `hid_add`. A unit whose norm stays flat near its (typically near-zero, see
# `initialize!`) starting value isn't acquiring any structure regardless of
# what its correlation/firing stats look like; a growing norm means the
# optimizer is actually shaping that unit's receptive field. Logged at iter=0
# (the pre-training baseline) and every LOG_EVERY iterations after.
function weight_norm_checkpoint(rbm, hid_add)
    w_add = rbm.w[:, :, hid_add]
    q, n_vis, h_add = size(w_add)
    wr = reshape(w_add, q * n_vis, h_add)
    norms = vec(sqrt.(sum(abs2, wr; dims=1)))
    return (
        norms     = Array(norms),
        mean_norm = mean(norms),
        max_norm  = maximum(norms),
        min_norm  = minimum(norms),
    )
end

function log_weight_norm_checkpoint(path_wn, iter, chk)
    open(path_wn, "a") do f
        println(f, "iter=$iter mean_norm=$(chk.mean_norm) max_norm=$(chk.max_norm) min_norm=$(chk.min_norm) norms=$(chk.norms)")
    end
    println("iter=$iter  weight_norm_check  mean=$(chk.mean_norm)  max=$(chk.max_norm)  min=$(chk.min_norm)")
end

# =============================================================================
# HELD-OUT PSEUDOLIKELIHOOD CHECKPOINT (does it generalize, or just memorize?)
# =============================================================================
# `lpl` (already logged) is the pseudolikelihood of the current training
# minibatch `vd` — it can improve throughout training even if the model is
# just memorizing the training set. This computes the same statistic on
# `data.XA_val`/`XB_val` (never trained on), so a gap that opens up between
# the two — training lpl improving while held-out lpl stalls or worsens — is
# the standard overfitting signal, and is what actually determines whether the
# correlations the model has learned (see V-V/A-B checks) can be trusted to
# generalize rather than being an artifact of this particular training set.
function log_heldout_checkpoint(path_lplval, iter, rbm_cpu, v_heldout_cpu)
    lpl_val = mean(log_pseudolikelihood(rbm_cpu, v_heldout_cpu))
    open(path_lplval, "a") do f; println(f, lpl_val); end
    println("iter=$iter  lpl_val=$lpl_val")
end

# =============================================================================
# FREE-BLOCK GRADIENT-NORM CHECKPOINT (is there still a learning signal here?)
# =============================================================================
# FREEZE CHECKPOINT's drift measures how hard Adam pushed on the *frozen*
# blocks before getting reset — useful for catching a broken freeze, but not
# for the free block, since the frozen-vs-reference comparison doesn't apply
# there. This instead recomputes the actual raw contrastive-divergence
# gradient — ∂d - ∂m from `∂free_energy`, the exact same quantity `pcd!`
# computes internally before regularization/optimizer state are applied — and
# reports its per-unit Frobenius norm restricted to the weight columns of
# `hid_add` (the free/added units for rbm_paired; all units for rbm_A/rbm_B,
# since nothing is frozen there). A norm that decays toward ~0 means the
# optimizer has run out of signal for that unit (converged, or stuck); one
# that stays flat-nonzero or grows means there's still real gradient driving
# it, independent of what the weight-norm or correlation checks show.
function free_gradient_checkpoint(rbm, vd, vm, hid_add)
    ∂d = ∂free_energy(rbm, vd)
    ∂m = ∂free_energy(rbm, vm)
    ∂w = (∂d.w .- ∂m.w)[:, :, hid_add]
    q, n_vis, h_add = size(∂w)
    gr = reshape(∂w, q * n_vis, h_add)
    norms = vec(sqrt.(sum(abs2, gr; dims=1)))
    return (
        norms     = Array(norms),
        mean_norm = mean(norms),
        max_norm  = maximum(norms),
        min_norm  = minimum(norms),
    )
end

function log_free_gradient_checkpoint(path_gn, iter, chk)
    open(path_gn, "a") do f
        println(f, "iter=$iter mean_norm=$(chk.mean_norm) max_norm=$(chk.max_norm) min_norm=$(chk.min_norm) norms=$(chk.norms)")
    end
    println("iter=$iter  free_grad_check  mean=$(chk.mean_norm)  max=$(chk.max_norm)  min=$(chk.min_norm)")
end

# =============================================================================
# TRAIN HELPER
# =============================================================================
function train_rbm!(rbm, X, iters, path_lpl; freeze_callback=nothing, path_freeze=nothing, rbm_A=nothing, rbm_B=nothing,
                     path_vh=nothing, hid_add=nothing, path_ab=nothing, vis_A=nothing, vis_B=nothing,
                     path_firing=nothing, path_v=nothing, path_wn=nothing,
                     path_lplval=nothing, v_heldout=nothing, path_gn=nothing)
    open(path_lpl, "w") do io end  # clear file
    check_freeze = !isnothing(freeze_callback) && !isnothing(path_freeze)
    check_freeze && (open(path_freeze, "w") do io end)
    check_vh = !isnothing(path_vh) && !isnothing(hid_add)
    check_vh && (open(path_vh, "w") do io end)
    check_ab = !isnothing(path_ab) && !isnothing(vis_A) && !isnothing(vis_B)
    check_ab && (open(path_ab, "w") do io end)
    check_firing = !isnothing(path_firing) && !isnothing(hid_add)
    check_firing && (open(path_firing, "w") do io end)
    check_v = !isnothing(path_v)
    check_v && (open(path_v, "w") do io end)
    check_wn = !isnothing(path_wn) && !isnothing(hid_add)
    if check_wn
        open(path_wn, "w") do io end
        log_weight_norm_checkpoint(path_wn, 0, weight_norm_checkpoint(rbm, hid_add))
    end
    check_lplval = !isnothing(path_lplval) && !isnothing(v_heldout)
    check_lplval && (open(path_lplval, "w") do io end)
    v_heldout_cpu = check_lplval ? cpu(v_heldout) : nothing
    check_gn = !isnothing(path_gn) && !isnothing(hid_add)
    check_gn && (open(path_gn, "w") do io end)
    # Gradient-norm clipping (applied before Adam sees the gradient): with
    # β1=0 (no momentum smoothing on the raw gradient) and a slow-moving
    # β2=0.999 second-moment estimate, a single large gradient spike late in
    # a long run can produce an enormous, poorly-normalized Adam step before
    # the second-moment average catches up — observed in practice as a
    # runaway weight-norm blowup in the added-unit block during long
    # (10,000-step) paired training at lr=1e-4, appearing in the final few
    # hundred steps with no recovery. Clipping the raw gradient's global norm
    # to CLIP_NORM before the Adam update removes the spike that triggers it.
    optim = OptimiserChain(ClipNorm(CLIP_NORM), Adam(Float32(LR), (0f0, 999f-3), 1f-6))
    @time pcd!(
        rbm, X;
        optim       = optim,
        iters       = iters,
        batchsize   = BATCH_SIZE,
        steps       = CD_STEPS,
        l2l1_weights = REG,
        ϵv=1f-1, ϵh=0f0, damping=1f-1, rescale_hidden=false,
        callback = function(; iter, vd, vm, kwargs...)
            if !isnothing(freeze_callback)
                if check_freeze && iszero(iter % LOG_EVERY)
                    drift_before = freeze_drift(rbm, rbm_A, rbm_B)
                    freeze_callback()
                    drift_after = freeze_drift(rbm, rbm_A, rbm_B)
                    log_freeze_checkpoint(path_freeze, iter, drift_before, drift_after)
                else
                    freeze_callback()
                end
            end
            if check_vh && iszero(iter % LOG_EVERY)
                chk = vh_learning_checkpoint(rbm, vd, vm, hid_add)
                log_vh_checkpoint(path_vh, iter, chk)
            end
            if check_ab && iszero(iter % LOG_EVERY)
                chk_ab = ab_correlation_checkpoint(vd, vm, vis_A, vis_B)
                log_ab_checkpoint(path_ab, iter, chk_ab)
            end
            if check_firing && iszero(iter % LOG_EVERY)
                chk_fr = firing_rate_checkpoint(rbm, vd, vm, hid_add)
                log_firing_checkpoint(path_firing, iter, chk_fr)
            end
            if check_v && iszero(iter % LOG_EVERY)
                chk_v = v_self_checkpoint(vd, vm)
                log_v_checkpoint(path_v, iter, chk_v)
            end
            if check_wn && iszero(iter % LOG_EVERY)
                log_weight_norm_checkpoint(path_wn, iter, weight_norm_checkpoint(rbm, hid_add))
            end
            if check_gn && iszero(iter % LOG_EVERY)
                log_free_gradient_checkpoint(path_gn, iter, free_gradient_checkpoint(rbm, vd, vm, hid_add))
            end
            if iszero(iter % LOG_EVERY)
                rbm_cpu = cpu(rbm)
                lpl = mean(log_pseudolikelihood(rbm_cpu, cpu(vd)))
                println("iter=$iter  lpl=$lpl")
                open(path_lpl, "a") do f; println(f, lpl); end
                if check_lplval
                    log_heldout_checkpoint(path_lplval, iter, rbm_cpu, v_heldout_cpu)
                end
            end
        end,
    )
end

# =============================================================================
# TRAIN INDIVIDUAL RBMs
# =============================================================================
# Skip retraining if this exact (N_HIDDEN, N_ITERS, REG, DATASET_TAG, SEED)
# private RBM already exists on disk — H_ADD-independent, so a sweep over
# H_ADD at fixed seed only needs to pay this cost once.
if isfile(path_rbm_A) && isfile(path_rbm_B)
    println("\n--- RBM A/B already trained for this seed — loading from disk ---")
    rbm_A = load_rbm(path_rbm_A)
    rbm_B = load_rbm(path_rbm_B)
    println("Loaded RBM A → $path_rbm_A")
    println("Loaded RBM B → $path_rbm_B")
else
    # rbm_A = RBM(PottsGumbel((q, N_VIS)), nsReLU((N_HIDDEN,)), zeros(q, N_VIS, N_HIDDEN))
    rbm_A = RBM(PottsGumbel((q, N_VIS)), xReLU((N_HIDDEN,)), zeros(q, N_VIS, N_HIDDEN))
    initialize!(rbm_A, XA)
    rbm_A = standardize(rbm_A)
    println("\n--- Training RBM A ---")
    rbm_A=gpu(rbm_A)
    train_rbm!(rbm_A, gpu(XA), N_ITERS, path_lpl_A; path_vh = path_vh_A, hid_add = 1:N_HIDDEN, path_v = path_v_A, path_wn = path_wn_A,
        path_lplval = path_lplval_A, v_heldout = gpu(XA_val), path_gn = path_gn_A)
    rbm_A=cpu(rbm_A)
    save_rbm_force(path_rbm_A, rbm_A)
    println("Saved RBM A → $path_rbm_A")
    println("VH-correlation checkpoints (RBM A) → $path_vh_A")
    println("V-correlation checkpoints (RBM A) → $path_v_A")
    println("Weight-norm checkpoints (RBM A) → $path_wn_A")
    println("Held-out pseudolikelihood checkpoints (RBM A) → $path_lplval_A")
    println("Free-gradient-norm checkpoints (RBM A) → $path_gn_A")

    # rbm_B = RBM(PottsGumbel((q, N_VIS)), nsReLU((N_HIDDEN,)), zeros(q, N_VIS, N_HIDDEN))
    rbm_B = RBM(PottsGumbel((q, N_VIS)), xReLU((N_HIDDEN,)), zeros(q, N_VIS, N_HIDDEN))
    initialize!(rbm_B, XB)
    rbm_B = standardize(rbm_B)
    println("\n--- Training RBM B ---")
    rbm_B=gpu(rbm_B)
    train_rbm!(rbm_B, gpu(XB), N_ITERS, path_lpl_B; path_vh = path_vh_B, hid_add = 1:N_HIDDEN, path_v = path_v_B, path_wn = path_wn_B,
        path_lplval = path_lplval_B, v_heldout = gpu(XB_val), path_gn = path_gn_B)
    rbm_B=cpu(rbm_B)
    save_rbm_force(path_rbm_B, rbm_B)
    println("Saved RBM B → $path_rbm_B")
    println("VH-correlation checkpoints (RBM B) → $path_vh_B")
    println("V-correlation checkpoints (RBM B) → $path_v_B")
    println("Weight-norm checkpoints (RBM B) → $path_wn_B")
    println("Held-out pseudolikelihood checkpoints (RBM B) → $path_lplval_B")
    println("Free-gradient-norm checkpoints (RBM B) → $path_gn_B")
end

# =============================================================================
# TRAIN PAIRED RBM
# =============================================================================
n_hid_total = 2 * N_HIDDEN + H_ADD
# PottsGumbel (not Potts) is required for GPU training: `Potts`'s sampling
# routine iterates element-by-element and hits CUDA's "scalar indexing is
# disallowed" error, whereas PottsGumbel uses the GPU-friendly Gumbel-softmax
# trick (same as rbm_A/rbm_B above). Both share the same `.par` layout, so the
# frozen-parameter projection is unaffected by this choice.
# rbm_paired = RBM(PottsGumbel((q, 2*N_VIS)), nsReLU((n_hid_total,)), zeros(q, 2*N_VIS, n_hid_total))
rbm_paired = RBM(PottsGumbel((q, 2*N_VIS)), xReLU((n_hid_total,)), zeros(q, 2*N_VIS, n_hid_total))
initialize!(rbm_paired, X_AB)
rbm_paired = standardize(rbm_paired)
project_to_frozen_par!(rbm_paired, rbm_A, rbm_B, H_ADD)

println("\n--- Training Paired RBM ---")
# rbm_A/rbm_B were moved back to cpu after their own training (for saving); the
# freeze reset runs every iteration against rbm_paired, so its reference RBMs
# must live on the same device (gpu) as rbm_paired, otherwise the projection's
# `.=` broadcast mixes CuArray and Array and blows up. Keep the cpu rbm_A/rbm_B
# untouched (already saved) and make separate gpu copies just for this reset.
rbm_A_gpu  = gpu(rbm_A)
rbm_B_gpu  = gpu(rbm_B)
rbm_paired = gpu(rbm_paired)
hid_add    = added_hidden_range(rbm_A_gpu, rbm_B_gpu, H_ADD)
(; vis_A, vis_B) = frozen_block_ranges(rbm_A_gpu, rbm_B_gpu)
train_rbm!(rbm_paired, gpu(X_AB), PAIRED_ITERS, path_lpl_paired;
    freeze_callback = () -> project_to_frozen_par!(rbm_paired, rbm_A_gpu, rbm_B_gpu, H_ADD),
    path_freeze = path_freeze_paired, rbm_A = rbm_A_gpu, rbm_B = rbm_B_gpu,
    path_vh = path_vh_paired, hid_add = hid_add,
    path_ab = path_ab_paired, vis_A = vis_A, vis_B = vis_B,
    path_firing = path_firing_paired, path_v = path_v_paired, path_wn = path_wn_paired,
    path_lplval = path_lplval_paired, v_heldout = gpu(X_AB_val), path_gn = path_gn_paired)
rbm_paired = cpu(rbm_paired)
save_rbm_force(path_rbm_paired, rbm_paired)
println("Saved paired RBM → $path_rbm_paired")
println("Freeze checkpoints → $path_freeze_paired")
println("VH-correlation checkpoints (added hidden units) → $path_vh_paired")
println("V-correlation checkpoints (whole visible layer) → $path_v_paired")
println("Firing-rate checkpoints (added hidden units) → $path_firing_paired")
println("A-B correlation checkpoints (cross-family structure) → $path_ab_paired")
println("Weight-norm checkpoints (added hidden units) → $path_wn_paired")
println("Held-out pseudolikelihood checkpoints (paired) → $path_lplval_paired")
println("Free-gradient-norm checkpoints (added hidden units) → $path_gn_paired")
