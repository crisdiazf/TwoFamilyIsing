using HDF5
using RestrictedBoltzmannMachines: Potts, nsReLU, RBM, initialize!, standardize, unstandardize
using RestrictedBoltzmannMachines: log_pseudolikelihood, pcd!, save_rbm
using RestrictedBoltzmannMachines: sample_h_from_v, sample_v_from_v
using Optimisers: Adam
using Statistics: mean
using Random
using TwoFamilyIsing
using OneHotArrays

Random.seed!(42)

# =============================================================================
# CONFIG
# =============================================================================
const DATA_PATH   = "./data/dataset_20links_09strength.bin"
const OUTPUT_DIR  = "./results"
isdir(OUTPUT_DIR) || mkdir(OUTPUT_DIR)

const N_HIDDEN    = 15
const H_ADD       = 10
const BATCH_SIZE  = 256
const CD_STEPS    = 30
const N_ITERS     = 10000
const PAIRED_ITERS = 3000
const REG         = 0.1
const LOG_EVERY   = 100

path_rbm_A      = joinpath(OUTPUT_DIR, "rbm_A.hdf5")
path_rbm_B      = joinpath(OUTPUT_DIR, "rbm_B.hdf5")
path_rbm_paired = joinpath(OUTPUT_DIR, "rbm_paired.hdf5")
path_lpl_A      = joinpath(OUTPUT_DIR, "lpl_A.txt")
path_lpl_B      = joinpath(OUTPUT_DIR, "lpl_B.txt")
path_lpl_paired = joinpath(OUTPUT_DIR, "lpl_paired.txt")

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

# =============================================================================
# PARAMETER PROJECTION
# =============================================================================
function project_to_frozen_par!(rbm_paired, rbm_A, rbm_B, h_add)
    n_vis_A, n_hid_A = size(rbm_A.w, 2), size(rbm_A.w, 3)
    n_vis_B, n_hid_B = size(rbm_B.w, 2), size(rbm_B.w, 3)
    n_vis_total = n_vis_A + n_vis_B

    vis_A = 1:n_vis_A
    vis_B = (n_vis_A + 1):n_vis_total
    hid_A = 1:n_hid_A
    hid_B = (n_hid_A + 1):(n_hid_A + n_hid_B)

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
# TRAIN HELPER
# =============================================================================
function train_rbm!(rbm, X, iters, path_lpl; freeze_callback=nothing)
    open(path_lpl, "w") do io end  # clear file
    @time pcd!(
        rbm, X;
        optim       = Adam(1f-4, (0f0, 999f-3), 1f-6),
        iters       = iters,
        batchsize   = BATCH_SIZE,
        steps       = CD_STEPS,
        l2l1_weights = REG,
        ϵv=1f-1, ϵh=0f0, damping=1f-1, rescale_hidden=false,
        callback = function(; iter, vd, kwargs...)
            isnothing(freeze_callback) || freeze_callback()
            if iszero(iter % LOG_EVERY)
                lpl = mean(log_pseudolikelihood(rbm, vd))
                println("iter=$iter  lpl=$lpl")
                open(path_lpl, "a") do f; println(f, lpl); end
            end
        end,
    )
end

# =============================================================================
# TRAIN INDIVIDUAL RBMs
# =============================================================================
rbm_A = RBM(Potts((q, N_VIS)), nsReLU((N_HIDDEN,)), zeros(q, N_VIS, N_HIDDEN))
initialize!(rbm_A, XA)
rbm_A = standardize(rbm_A)
println("\n--- Training RBM A ---")
train_rbm!(rbm_A, XA, N_ITERS, path_lpl_A)
save_rbm(path_rbm_A, rbm_A)
println("Saved RBM A → $path_rbm_A")

rbm_B = RBM(Potts((q, N_VIS)), nsReLU((N_HIDDEN,)), zeros(q, N_VIS, N_HIDDEN))
initialize!(rbm_B, XB)
rbm_B = standardize(rbm_B)
println("\n--- Training RBM B ---")
train_rbm!(rbm_B, XB, N_ITERS, path_lpl_B)
save_rbm(path_rbm_B, rbm_B)
println("Saved RBM B → $path_rbm_B")

# =============================================================================
# TRAIN PAIRED RBM
# =============================================================================
n_hid_total = 2 * N_HIDDEN + H_ADD
rbm_paired = RBM(Potts((q, 2*N_VIS)), nsReLU((n_hid_total,)), zeros(q, 2*N_VIS, n_hid_total))
initialize!(rbm_paired, X_AB)
rbm_paired = standardize(rbm_paired)
project_to_frozen_par!(rbm_paired, rbm_A, rbm_B, H_ADD)

println("\n--- Training Paired RBM ---")
train_rbm!(rbm_paired, X_AB, PAIRED_ITERS, path_lpl_paired;
    freeze_callback = () -> project_to_frozen_par!(rbm_paired, rbm_A, rbm_B, H_ADD))
save_rbm(path_rbm_paired, rbm_paired)
println("Saved paired RBM → $path_rbm_paired")
