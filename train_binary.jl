using CUDA
using HDF5
using RestrictedBoltzmannMachines: BinaryRBM, RBM, initialize!, standardize
using RestrictedBoltzmannMachines: log_pseudolikelihood, pcd!, save_rbm
using RestrictedBoltzmannMachines: sample_h_from_v, sample_v_from_v, mean_h_from_v, gpu, cpu
using RestrictedBoltzmannMachines: ∂free_energy
using Optimisers: Adam
using Statistics: mean
using Random
using TwoFamilyIsing

Random.seed!(42)

# =============================================================================
# CONFIG
# =============================================================================
# Same architecture/workflow as train_potts.jl (private RBM_A/RBM_B, frozen
# blocks projected into a paired RBM with H_ADD free hidden units, same set of
# checkpoints), but using BinaryRBM (binary visible AND binary hidden units)
# instead of Potts(q=2)/PottsGumbel visible + nsReLU hidden. Since the data is
# already binary (Ising spins mapped to 0/1 states), this drops the one-hot
# encoding entirely: visible arrays are plain (n_vis, batch), not (q, n_vis, batch).
# File names are prefixed with "binary_" so results sit alongside train_potts.jl's
# output in the same directory for direct comparison.
# ARGS: N_ITERS PAIRED_ITERS [H_ADD=30] [N_HIDDEN=30] [REG=0] [DATASET_TAG=20links_09strength]
const N_ITERS      = parse(Int, ARGS[1])
const PAIRED_ITERS = parse(Int, ARGS[2])
const H_ADD        = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 30
const N_HIDDEN     = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 30
const REG          = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 0
const DATASET_TAG  = length(ARGS) >= 6 ? ARGS[6] : "20links_09strength"
const DATA_PATH    = "./data/dataset_$(DATASET_TAG).bin"
const OUTPUT_DIR   = "./results"
isdir(OUTPUT_DIR) || mkdir(OUTPUT_DIR)

const BATCH_SIZE  = 512
const CD_STEPS    = 50
const LOG_EVERY   = 100

regstr(r) = isinteger(r) ? string(Int(r)) : string(r)  # REG=0 (not 0.0) matches pre-existing filenames
single_suffix() = "N_HIDDEN=$(N_HIDDEN)_H_ADD=$(H_ADD)_K=$(CD_STEPS)_N_ITERS=$(N_ITERS)_REG=$(regstr(REG))_DATA=$(DATASET_TAG)"
paired_suffix()  = single_suffix() * "_PAIRED_ITERS=$(PAIRED_ITERS)"

path_rbm_A      = joinpath(OUTPUT_DIR, "binary_rbm_A_$(single_suffix()).hdf5")
path_rbm_B      = joinpath(OUTPUT_DIR, "binary_rbm_B_$(single_suffix()).hdf5")
path_rbm_paired = joinpath(OUTPUT_DIR, "binary_rbm_paired_$(paired_suffix()).hdf5")
path_lpl_A      = joinpath(OUTPUT_DIR, "binary_lpl_A_$(single_suffix()).txt")
path_lpl_B      = joinpath(OUTPUT_DIR, "binary_lpl_B_$(single_suffix()).txt")
path_vh_A       = joinpath(OUTPUT_DIR, "binary_vh_check_A_$(single_suffix()).txt")
path_vh_B       = joinpath(OUTPUT_DIR, "binary_vh_check_B_$(single_suffix()).txt")
path_vv_A       = joinpath(OUTPUT_DIR, "binary_vv_check_A_$(single_suffix()).txt")
path_vv_B       = joinpath(OUTPUT_DIR, "binary_vv_check_B_$(single_suffix()).txt")
path_wn_A       = joinpath(OUTPUT_DIR, "binary_wn_check_A_$(single_suffix()).txt")
path_wn_B       = joinpath(OUTPUT_DIR, "binary_wn_check_B_$(single_suffix()).txt")
path_lpl_paired = joinpath(OUTPUT_DIR, "binary_lpl_paired_$(paired_suffix()).txt")
path_freeze_paired = joinpath(OUTPUT_DIR, "binary_freeze_check_paired_$(paired_suffix()).txt")
path_vh_paired  = joinpath(OUTPUT_DIR, "binary_vh_check_paired_$(paired_suffix()).txt")
path_ab_paired  = joinpath(OUTPUT_DIR, "binary_ab_check_paired_$(paired_suffix()).txt")
path_firing_paired = joinpath(OUTPUT_DIR, "binary_firing_check_paired_$(paired_suffix()).txt")
path_vv_paired  = joinpath(OUTPUT_DIR, "binary_vv_check_paired_$(paired_suffix()).txt")
path_wn_paired  = joinpath(OUTPUT_DIR, "binary_wn_check_paired_$(paired_suffix()).txt")
path_lplval_A       = joinpath(OUTPUT_DIR, "binary_lplval_A_$(single_suffix()).txt")
path_lplval_B       = joinpath(OUTPUT_DIR, "binary_lplval_B_$(single_suffix()).txt")
path_lplval_paired  = joinpath(OUTPUT_DIR, "binary_lplval_paired_$(paired_suffix()).txt")
path_gn_A       = joinpath(OUTPUT_DIR, "binary_gn_check_A_$(single_suffix()).txt")
path_gn_B       = joinpath(OUTPUT_DIR, "binary_gn_check_B_$(single_suffix()).txt")
path_gn_paired  = joinpath(OUTPUT_DIR, "binary_gn_check_paired_$(paired_suffix()).txt")

# =============================================================================
# DATA
# =============================================================================
# No one-hot encoding: Binary visible units take the 0/1 states directly as a
# plain (n_vis, batch) array (vs. Potts's (q, n_vis, batch) one-hot tensor).
data     = load_dataset(DATA_PATH)
XA_spins = data.XA_train'
XB_spins = data.XB_train'

XA = @. Float32((XA_spins + 1) / 2)
XB = @. Float32((XB_spins + 1) / 2)
X_AB = vcat(XA, XB)

N_VIS = size(XA, 1)

println("XA: $(size(XA))  XB: $(size(XB))  X_AB: $(size(X_AB))")

# Held-out validation split (data.XA_val/XB_val), never trained on. Used only
# to check that the pseudolikelihood improvement seen on `vd` (a training
# minibatch) actually generalizes, rather than just reflecting memorization —
# see the HELD-OUT PSEUDOLIKELIHOOD CHECKPOINT section below.
XA_val = @. Float32((data.XA_val' + 1) / 2)
XB_val = @. Float32((data.XB_val' + 1) / 2)
X_AB_val = vcat(XA_val, XB_val)

println("XA_val: $(size(XA_val))  XB_val: $(size(XB_val))  X_AB_val: $(size(X_AB_val))")

# =============================================================================
# PARAMETER PROJECTION
# =============================================================================
function frozen_block_ranges(rbm_A, rbm_B)
    n_vis_A, n_hid_A = size(rbm_A.w, 1), size(rbm_A.w, 2)
    n_vis_B, n_hid_B = size(rbm_B.w, 1), size(rbm_B.w, 2)
    n_vis_total = n_vis_A + n_vis_B

    vis_A = 1:n_vis_A
    vis_B = (n_vis_A + 1):n_vis_total
    hid_A = 1:n_hid_A
    hid_B = (n_hid_A + 1):(n_hid_A + n_hid_B)
    return (; vis_A, vis_B, hid_A, hid_B)
end

function project_to_frozen_par!(rbm_paired, rbm_A, rbm_B, h_add)
    (; vis_A, vis_B, hid_A, hid_B) = frozen_block_ranges(rbm_A, rbm_B)

    rbm_paired.visible.par[:, vis_A] .= rbm_A.visible.par
    rbm_paired.visible.par[:, vis_B] .= rbm_B.visible.par
    rbm_paired.hidden.par[:, hid_A]  .= rbm_A.hidden.par
    rbm_paired.hidden.par[:, hid_B]  .= rbm_B.hidden.par

    rbm_paired.w[vis_A, hid_A] .= rbm_A.w
    rbm_paired.w[vis_A, hid_B] .= 0
    rbm_paired.w[vis_B, hid_A] .= 0
    rbm_paired.w[vis_B, hid_B] .= rbm_B.w
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
        vis_A = maximum(abs, rbm_paired.visible.par[:, vis_A] .- rbm_A.visible.par),
        vis_B = maximum(abs, rbm_paired.visible.par[:, vis_B] .- rbm_B.visible.par),
        hid_A = maximum(abs, rbm_paired.hidden.par[:, hid_A] .- rbm_A.hidden.par),
        hid_B = maximum(abs, rbm_paired.hidden.par[:, hid_B] .- rbm_B.hidden.par),
        w_AA  = maximum(abs, rbm_paired.w[vis_A, hid_A] .- rbm_A.w),
        w_BB  = maximum(abs, rbm_paired.w[vis_B, hid_B] .- rbm_B.w),
        w_AB  = maximum(abs, rbm_paired.w[vis_A, hid_B]),  # cross block, must stay 0
        w_BA  = maximum(abs, rbm_paired.w[vis_B, hid_A]),  # cross block, must stay 0
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
# converged, or because it is "dead" (e.g. stuck at the same activation for
# both data and fantasy particles). We track this specifically for the H_ADD
# extra hidden units in rbm_paired, since those are the only ones actually
# free to learn (see FREEZE CHECKPOINT above), and for all units of rbm_A/rbm_B.
function added_hidden_range(rbm_A, rbm_B, h_add)
    n_hid_A = size(rbm_A.w, 2)
    n_hid_B = size(rbm_B.w, 2)
    return (n_hid_A + n_hid_B + 1):(n_hid_A + n_hid_B + h_add)
end

# <v h> averaged over the batch: v is (n_vis, batch), h is (n_hid, batch);
# returns (n_vis, n_hid). No one-hot/q dimension to reshape away, unlike Potts.
function vh_correlation(v, h)
    batch = size(v, 2)
    return (Float32.(v) * h') ./ batch
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
# For binary visible units, the connected correlation between site i and site j is
#   C[i,j] = <v_i v_j> - <v_i><v_j>
# computed once under the data distribution (from vd) and once under the
# model (from the persistent fantasy chains vm), restricted to i ∈ family A,
# j ∈ family B. In a well-trained joint model these two tensors should align
# along y=x (r → 1, slope → 1, intercept → 0), exactly like the vh check above.
function vv_connected_correlation(v_i, v_j)
    batch = size(v_i, 2)
    vi = Float32.(v_i)
    vj = Float32.(v_j)
    mean_i = vec(mean(vi; dims=2))
    mean_j = vec(mean(vj; dims=2))
    cross  = (vi * vj') ./ batch
    return cross .- mean_i * mean_j'
end

function ab_correlation_checkpoint(vd, vm, vis_A, vis_B)
    c_data  = vv_connected_correlation(vd[vis_A, :], vd[vis_B, :])
    c_model = vv_connected_correlation(vm[vis_A, :], vm[vis_B, :])
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
# V-V CORRELATION CHECKPOINT (does the model itself reproduce the data's
# pairwise structure — the actual inverse-Ising target, not just a proxy?)
# =============================================================================
# The vh check only tells you whether the hidden units are receiving/aligning
# with a training signal; it says nothing about whether the visible layer
# they're built on actually reproduces the data's site-site correlations —
# which, for an Ising-type model, is the real object of interest. This
# computes the same connected-correlation/linear-alignment diagnostic as the
# A-B check, but for *all* pairs of distinct visible sites within a single
# RBM's own visible layer (same-site "self-correlation" pairs i==j are
# excluded — they're basically guaranteed to align since the single-site
# fields fit the marginals directly, and would inflate r/slope with a trivial
# signal). Applies equally to rbm_A, rbm_B (their whole visible layer) and to
# rbm_paired (its whole 2*N_VIS visible layer, i.e. intra-A + intra-B + A-B
# combined — a superset of what ab_correlation_checkpoint targets).
function vv_offdiag_vec(C)
    n_vis = size(C, 1)
    Ccpu = Array(C)
    vals = Vector{Float32}(undef, n_vis * (n_vis - 1))
    k = 0
    for j in 1:n_vis, i in 1:n_vis
        i == j && continue
        k += 1
        vals[k] = Ccpu[i, j]
    end
    return vals
end

function vv_self_correlation_checkpoint(vd, vm)
    c_data  = vv_connected_correlation(vd, vd)
    c_model = vv_connected_correlation(vm, vm)
    x = vv_offdiag_vec(c_data)
    y = vv_offdiag_vec(c_model)
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

function log_vv_checkpoint(path_vv, iter, chk)
    open(path_vv, "a") do f
        println(f, "iter=$iter mean_abs_diff=$(chk.mean_abs_diff) max_abs_diff=$(chk.max_abs_diff) r=$(chk.r) slope=$(chk.slope) intercept=$(chk.intercept)")
    end
    println("iter=$iter  vv_check  mean|C_data-C_model|=$(chk.mean_abs_diff)  r=$(chk.r)  slope=$(chk.slope)  intercept=$(chk.intercept)")
end

# =============================================================================
# FIRING-RATE CHECKPOINT (per added-unit activity, distinct from alignment)
# =============================================================================
# Binary hidden units are exactly 0/1, so this is an exact P(unit=1 | v),
# estimated from real samples (not mean-field values) of data (vd) and model
# (vm). r/slope in the vh check are computed from mean activations, so a unit
# that's saturated — firing on nearly every sample or almost none, regardless
# of the input — can still show up with decent-looking alignment stats. Firing
# rate exposes that directly: a unit stuck near 0 or 1 for both data and
# model, and staying that way across checkpoints, isn't discriminating between
# inputs — regardless of what its correlation stats say.
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
# norm of the incoming visible weight column, ‖w[:, j]‖, for each unit in
# `hid_add`. A unit whose norm stays flat near its (typically near-zero, see
# `initialize!`) starting value isn't acquiring any structure regardless of
# what its correlation/firing stats look like; a growing norm means the
# optimizer is actually shaping that unit's receptive field. Logged at iter=0
# (the pre-training baseline) and every LOG_EVERY iterations after.
function weight_norm_checkpoint(rbm, hid_add)
    w_add = rbm.w[:, hid_add]
    norms = vec(sqrt.(sum(abs2, w_add; dims=1)))
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
    ∂w = (∂d.w .- ∂m.w)[:, hid_add]
    norms = vec(sqrt.(sum(abs2, ∂w; dims=1)))
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
                     path_firing=nothing, path_vv=nothing, path_wn=nothing,
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
    check_vv = !isnothing(path_vv)
    check_vv && (open(path_vv, "w") do io end)
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
    @time pcd!(
        rbm, X;
        optim       = Adam(1f-3, (0f0, 999f-3), 1f-6),
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
            if check_vv && iszero(iter % LOG_EVERY)
                chk_vv = vv_self_correlation_checkpoint(vd, vm)
                log_vv_checkpoint(path_vv, iter, chk_vv)
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
rbm_A = BinaryRBM(Float32, N_VIS, N_HIDDEN)
initialize!(rbm_A, XA)
rbm_A = standardize(rbm_A)
println("\n--- Training RBM A (Binary) ---")
rbm_A=gpu(rbm_A)
train_rbm!(rbm_A, gpu(XA), N_ITERS, path_lpl_A; path_vh = path_vh_A, hid_add = 1:N_HIDDEN, path_vv = path_vv_A, path_wn = path_wn_A,
    path_lplval = path_lplval_A, v_heldout = gpu(XA_val), path_gn = path_gn_A)
rbm_A=cpu(rbm_A)
save_rbm(path_rbm_A, rbm_A)
println("Saved RBM A → $path_rbm_A")
println("VH-correlation checkpoints (RBM A) → $path_vh_A")
println("VV-correlation checkpoints (RBM A) → $path_vv_A")
println("Weight-norm checkpoints (RBM A) → $path_wn_A")
println("Held-out pseudolikelihood checkpoints (RBM A) → $path_lplval_A")
println("Free-gradient-norm checkpoints (RBM A) → $path_gn_A")

rbm_B = BinaryRBM(Float32, N_VIS, N_HIDDEN)
initialize!(rbm_B, XB)
rbm_B = standardize(rbm_B)
println("\n--- Training RBM B (Binary) ---")
rbm_B=gpu(rbm_B)
train_rbm!(rbm_B, gpu(XB), N_ITERS, path_lpl_B; path_vh = path_vh_B, hid_add = 1:N_HIDDEN, path_vv = path_vv_B, path_wn = path_wn_B,
    path_lplval = path_lplval_B, v_heldout = gpu(XB_val), path_gn = path_gn_B)
rbm_B=cpu(rbm_B)
save_rbm(path_rbm_B, rbm_B)
println("Saved RBM B → $path_rbm_B")
println("VH-correlation checkpoints (RBM B) → $path_vh_B")
println("VV-correlation checkpoints (RBM B) → $path_vv_B")
println("Weight-norm checkpoints (RBM B) → $path_wn_B")
println("Held-out pseudolikelihood checkpoints (RBM B) → $path_lplval_B")
println("Free-gradient-norm checkpoints (RBM B) → $path_gn_B")

# =============================================================================
# TRAIN PAIRED RBM
# =============================================================================
n_hid_total = 2 * N_HIDDEN + H_ADD
rbm_paired = BinaryRBM(Float32, 2 * N_VIS, n_hid_total)
initialize!(rbm_paired, X_AB)
rbm_paired = standardize(rbm_paired)
project_to_frozen_par!(rbm_paired, rbm_A, rbm_B, H_ADD)

println("\n--- Training Paired RBM (Binary) ---")
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
    path_firing = path_firing_paired, path_vv = path_vv_paired, path_wn = path_wn_paired,
    path_lplval = path_lplval_paired, v_heldout = gpu(X_AB_val), path_gn = path_gn_paired)
rbm_paired = cpu(rbm_paired)
save_rbm(path_rbm_paired, rbm_paired)
println("Saved paired RBM → $path_rbm_paired")
println("Freeze checkpoints → $path_freeze_paired")
println("VH-correlation checkpoints (added hidden units) → $path_vh_paired")
println("VV-correlation checkpoints (whole visible layer) → $path_vv_paired")
println("Firing-rate checkpoints (added hidden units) → $path_firing_paired")
println("A-B correlation checkpoints (cross-family structure) → $path_ab_paired")
println("Weight-norm checkpoints (added hidden units) → $path_wn_paired")
println("Held-out pseudolikelihood checkpoints (paired) → $path_lplval_paired")
println("Free-gradient-norm checkpoints (added hidden units) → $path_gn_paired")
