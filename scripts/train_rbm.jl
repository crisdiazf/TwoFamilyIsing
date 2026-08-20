# # Paired RBM Analysis for Coupled Spin Systems
#
# This notebook analyzes coupled spin systems using Restricted Boltzmann Machines,
# training individual RBMs and then combining them into a paired architecture.
# Includes comprehensive validation tests for model quality and statistics preservation.

# ## Environment Setup

using CairoMakie
using Makie
using TwoFamilyIsing
using HDF5
using RestrictedBoltzmannMachines: RBM, BinaryRBM, initialize!
using RestrictedBoltzmannMachines: log_pseudolikelihood
using RestrictedBoltzmannMachines: pcd!
using RestrictedBoltzmannMachines: sample_from_inputs
using RestrictedBoltzmannMachines: sample_v_from_v
using RestrictedBoltzmannMachines: save_rbm, load_rbm
using RestrictedBoltzmannMachines: free_energy
using RestrictedBoltzmannMachines: sample_h_from_v, sample_v_from_h
using Statistics: mean, std, var, cor
using ValueHistories: @trace, MVHistory
using Random
using Plots: heatmap, scatter
using LinearAlgebra: diagm, diag
using Plots

# Set random seed for reproducibility
Random.seed!(42)

# ## Data Loading and Preprocessing

# Load the example dataset
const DATA_PATH = "./data/dataset_20links_09strength.bin"
data = load_dataset(DATA_PATH)

# Convert spin variables {-1,1} to binary {0,1}
XA_spins = data.XA_train'
XB_spins = data.XB_train'

XA = @. (XA_spins + 1) / 2
XB = @. (XB_spins + 1) / 2

println("Data shapes - XA: $(size(XA)), XB: $(size(XB))")

# ## Individual RBM Training

# ### RBM Configuration
const N_VISIBLE = 30
const N_HIDDEN = 15
const BATCH_SIZE = 256
const N_ITERATIONS = 30000
const CD_STEPS = 30

# ### Train RBM for System A
println("\n" * "="^50)
println("Training RBM A")
println("="^50)

rbm_A = BinaryRBM(Float32, (N_VISIBLE,), N_HIDDEN)
initialize!(rbm_A, XA)

# Initial log-pseudolikelihood
println("Initial log(PL) for RBM A: ", mean(@time log_pseudolikelihood(rbm_A, XA)))

# Training with progress tracking
history_A = MVHistory()
@time pcd!(
    rbm_A, XA; 
    iters=N_ITERATIONS, 
    batchsize=BATCH_SIZE, 
    steps=CD_STEPS,
    callback = function(; iter, _...)
        if iszero(iter % 100)
            lpl = mean(log_pseudolikelihood(rbm_A, XA))
            println("iter: $iter lpl=$lpl")
            @trace history_A iter lpl
        end
    end
)

# ### Visualize Training Progress for RBM A
fig = Makie.Figure(resolution=(500, 300))
ax = Makie.Axis(fig[1, 1], 
    xlabel="Iteration", 
    ylabel="Log-Pseudolikelihood",
    title="RBM A Training Progress")
Makie.lines!(ax, get(history_A, :lpl)...)
fig

# ### Train RBM for System B
println("\n" * "="^50)
println("Training RBM B")
println("="^50)

rbm_B = BinaryRBM(Float32, (N_VISIBLE,), N_HIDDEN)
initialize!(rbm_B, XB)

# Initial log-pseudolikelihood
println("Initial log(PL) for RBM B: ", mean(@time log_pseudolikelihood(rbm_B, XB)))

# Training with progress tracking
history_B = MVHistory()
@time pcd!(
    rbm_B, XB; 
    iters=N_ITERATIONS, 
    batchsize=BATCH_SIZE, 
    steps=CD_STEPS,
    callback = function(; iter, _...)
        if iszero(iter % 100)
            lpl = mean(log_pseudolikelihood(rbm_B, XB))
            println("iter: $iter lpl=$lpl")
            @trace history_B iter lpl
        end
    end
)

# ### Visualize Training Progress for RBM B
fig = Makie.Figure(resolution=(500, 300))
ax = Makie.Axis(fig[1, 1], 
    xlabel="Iteration", 
    ylabel="Log-Pseudolikelihood",
    title="RBM B Training Progress")
Makie.lines!(ax, get(history_B, :lpl)...)
fig

# ### Save Trained RBMs
const RBM_A_PATH = "./rbmA_k=30_tt=30k_dataset_20links_09strength.hdf5"
const RBM_B_PATH = "./rbmB_k=30_tt=30k_dataset_20links_09strength.hdf5"
save_rbm(RBM_A_PATH, rbm_A)
save_rbm(RBM_B_PATH, rbm_B)
println("Saved RBM A to $RBM_A_PATH")
println("Saved RBM B to $RBM_B_PATH")

# ## Free Energy Analysis for Individual RBMs

# Sample and track free energy during Gibbs sampling
const N_SAMPLES = 1000
const N_GIBBS_STEPS = 30

# ### Free Energy for RBM A
fantasy_F_A = zeros(N_SAMPLES, N_GIBBS_STEPS)
fantasy_x_A = bitrand(N_VISIBLE, N_SAMPLES)
fantasy_F_A[:, 1] .= free_energy(rbm_A, fantasy_x_A)

println("\nRunning Gibbs sampling for RBM A...")
@time for t in 2:N_GIBBS_STEPS
    fantasy_x_A .= sample_v_from_v(rbm_A, fantasy_x_A, steps=50)
    fantasy_F_A[:, t] .= free_energy(rbm_A, fantasy_x_A)
    println("step $t")
end

# ### Free Energy for RBM B
fantasy_F_B = zeros(N_SAMPLES, N_GIBBS_STEPS)
fantasy_x_B = bitrand(N_VISIBLE, N_SAMPLES)
fantasy_F_B[:, 1] .= free_energy(rbm_B, fantasy_x_B)

println("\nRunning Gibbs sampling for RBM B...")
@time for t in 2:N_GIBBS_STEPS
    fantasy_x_B .= sample_v_from_v(rbm_B, fantasy_x_B, steps=50)
    fantasy_F_B[:, t] .= free_energy(rbm_B, fantasy_x_B)
    println("step $t")
end

# ### Visualize Free Energy Convergence
fig = Makie.Figure(resolution=(800, 300))

# RBM A
ax1 = Makie.Axis(fig[1, 1], 
    xlabel="Sampling Step", 
    ylabel="Free Energy",
    title="Free Energy Convergence (RBM A)")
fantasy_F_μ_A = vec(mean(fantasy_F_A; dims=1))
fantasy_F_σ_A = vec(std(fantasy_F_A; dims=1))
Makie.band!(ax1, 1:N_GIBBS_STEPS, 
    fantasy_F_μ_A - fantasy_F_σ_A/2, 
    fantasy_F_μ_A + fantasy_F_σ_A/2)
Makie.lines!(ax1, 1:N_GIBBS_STEPS, fantasy_F_μ_A)

# RBM B
ax2 = Makie.Axis(fig[1, 2], 
    xlabel="Sampling Step", 
    ylabel="Free Energy",
    title="Free Energy Convergence (RBM B)")
fantasy_F_μ_B = vec(mean(fantasy_F_B; dims=1))
fantasy_F_σ_B = vec(std(fantasy_F_B; dims=1))
Makie.band!(ax2, 1:N_GIBBS_STEPS, 
    fantasy_F_μ_B - fantasy_F_σ_B/2, 
    fantasy_F_μ_B + fantasy_F_σ_B/2)
Makie.lines!(ax2, 1:N_GIBBS_STEPS, fantasy_F_μ_B)

fig

# ## Model Validation Tests for Individual RBMs

# ### Helper function for activation comparison plots
function plot_activation_comparison(data_mean, model_mean, title, xlabel, ylabel)
    """Create scatter plot comparing data vs model activations with y=x line."""
    p = Plots.scatter(data_mean, model_mean, 
        title=title * " (corr=$(round(cor(data_mean, model_mean), digits=4)))",
        xlabel=xlabel, ylabel=ylabel,
        label="", alpha=0.6, markersize=3)
    Plots.plot!(identity, color=:red, linestyle=:dash, label="y = x")
    return p
end

# ### Test 1: Hidden unit activations
println("\n" * "="^50)
println("Validation: Hidden Unit Activations")
println("="^50)

# Get hidden activations for data and model samples
h_data_A = sample_h_from_v(rbm_A, XA) .|> Float64
h_model_A = sample_h_from_v(rbm_A, fantasy_x_A) .|> Float64
h_data_B = sample_h_from_v(rbm_B, XB) .|> Float64
h_model_B = sample_h_from_v(rbm_B, fantasy_x_B) .|> Float64

# Compute mean activations
h_data_mean_A = vec(mean(h_data_A; dims=2))
h_model_mean_A = vec(mean(h_model_A; dims=2))
h_data_mean_B = vec(mean(h_data_B; dims=2))
h_model_mean_B = vec(mean(h_model_B; dims=2))

# Plot hidden activation comparisons
p1 = plot_activation_comparison(h_data_mean_A, h_model_mean_A, 
    "Hidden Activations - RBM A", "⟨h⟩ data", "⟨h⟩ model")
p2 = plot_activation_comparison(h_data_mean_B, h_model_mean_B, 
    "Hidden Activations - RBM B", "⟨h⟩ data", "⟨h⟩ model")
Plots.plot(p1, p2, layout=(1, 2), size=(800, 400))

# ### Test 2: Visible unit activations
println("\n" * "="^50)
println("Validation: Visible Unit Activations")
println("="^50)

v_data_mean_A = vec(mean(XA; dims=2))
v_model_mean_A = vec(mean(fantasy_x_A; dims=2))
v_data_mean_B = vec(mean(XB; dims=2))
v_model_mean_B = vec(mean(fantasy_x_B; dims=2))

p3 = plot_activation_comparison(v_data_mean_A, v_model_mean_A, 
    "Visible Activations - RBM A", "⟨v⟩ data", "⟨v⟩ model")
p4 = plot_activation_comparison(v_data_mean_B, v_model_mean_B, 
    "Visible Activations - RBM B", "⟨v⟩ data", "⟨v⟩ model")
Plots.plot(p3, p4, layout=(1, 2), size=(800, 400))

# ### Test 3: Hidden-Visible correlations
println("\n" * "="^50)
println("Validation: H-V Correlations")
println("="^50)

# Compute correlation matrices
hv_data_A = h_data_A * XA' / size(XA, 2)
hv_model_A = h_model_A * fantasy_x_A' / size(fantasy_x_A, 2)
hv_data_B = h_data_B * XB' / size(XB, 2)
hv_model_B = h_model_B * fantasy_x_B' / size(fantasy_x_B, 2)

# Flatten and compare
hv_data_flat_A = vec(hv_data_A)
hv_model_flat_A = vec(hv_model_A)
hv_data_flat_B = vec(hv_data_B)
hv_model_flat_B = vec(hv_model_B)

p5 = plot_activation_comparison(hv_data_flat_A, hv_model_flat_A, 
    "H-V Correlations - RBM A", "⟨hv⟩ data", "⟨hv⟩ model")
p6 = plot_activation_comparison(hv_data_flat_B, hv_model_flat_B, 
    "H-V Correlations - RBM B", "⟨hv⟩ data", "⟨hv⟩ model")
Plots.plot(p5, p6, layout=(1, 2), size=(800, 400))

# ## Correlation Analysis for Individual RBMs

# Compare empirical correlations
CA_model = empirical_corr(fantasy_x_A')
CA_data = empirical_corr(XA')
CB_model = empirical_corr(fantasy_x_B')
CB_data = empirical_corr(XB')

# Visualize correlation matrices
heatmap(CA_model, title="Model Correlations (RBM A)")
heatmap(CA_data, title="Data Correlations (System A)")
heatmap(CB_model, title="Model Correlations (RBM B)")
heatmap(CB_data, title="Data Correlations (System B)")

# ## Paired RBM Architecture
#
# We now combine the two systems into a paired RBM architecture
# with additional hidden units to capture cross-system interactions.

# Combine datasets
X_AB = vcat(XA, XB)
println("Combined dataset shape: $(size(X_AB))")

# ### Initialize Paired RBM
const H_ADD = 10  # Additional hidden units for cross-interactions
const N_VIS_TOTAL = 60
const N_HID_TOTAL = 30 + H_ADD

rbm_paired = BinaryRBM(Float32, (N_VIS_TOTAL,), N_HID_TOTAL)
initialize!(rbm_paired, X_AB)

# ### Project Frozen Parameters
"""
    project_to_frozen_par!(rbm_paired, rbm_A, rbm_B, h_add)

Project parameters from two individual RBMs into a paired RBM structure.

The paired RBM architecture:
- Visible units: first n_vis_A from system A, remaining from system B
- Hidden units: first n_hid_A from rbm_A, next n_hid_B from rbm_B, 
                and h_add extra trainable units at the end

Frozen connections:
- rbm_A visible → rbm_A hidden (frozen)
- rbm_B visible → rbm_B hidden (frozen)
- No cross connections between A and B (frozen to 0)
- Connections involving h_add units remain trainable
"""
function project_to_frozen_par!(rbm_paired, rbm_A, rbm_B, h_add)
    # Get dimensions
    n_vis_A = size(rbm_A.w, 1)
    n_hid_A = size(rbm_A.w, 2)
    n_vis_B = size(rbm_B.w, 1)
    n_hid_B = size(rbm_B.w, 2)
    
    # Total dimensions
    n_vis_total = n_vis_A + n_vis_B
    n_hid_total = n_hid_A + n_hid_B + h_add
    
    # Index ranges
    vis_A_range = 1:n_vis_A
    vis_B_range = (n_vis_A + 1):n_vis_total
    hid_A_range = 1:n_hid_A
    hid_B_range = (n_hid_A + 1):(n_hid_A + n_hid_B)
    hid_add_range = (n_hid_A + n_hid_B + 1):n_hid_total
    
    # Project visible biases
    rbm_paired.visible.par[:, vis_A_range] .= rbm_A.visible.par
    rbm_paired.visible.par[:, vis_B_range] .= rbm_B.visible.par
    
    # Project hidden biases (frozen for A and B, trainable for additional)
    rbm_paired.hidden.par[:, hid_A_range] .= rbm_A.hidden.par
    rbm_paired.hidden.par[:, hid_B_range] .= rbm_B.hidden.par
    # hid_add_range biases remain as initialized (trainable)
    
    # Project weight matrix
    # Block (A → A): frozen
    rbm_paired.w[vis_A_range, hid_A_range] .= rbm_A.w
    
    # Block (A → B): frozen to zero
    rbm_paired.w[vis_A_range, hid_B_range] .= 0
    
    # Block (B → A): frozen to zero
    rbm_paired.w[vis_B_range, hid_A_range] .= 0
    
    # Block (B → B): frozen
    rbm_paired.w[vis_B_range, hid_B_range] .= rbm_B.w
    
    # Connections involving h_add remain as initialized (trainable)
    
    return rbm_paired
end

# Apply parameter projection
project_to_frozen_par!(rbm_paired, rbm_A, rbm_B, H_ADD)

# ## Paired RBM Training

# Initial evaluation
println("\nInitial log(PL) for paired RBM: ", 
    mean(@time log_pseudolikelihood(rbm_paired, X_AB)))

# Train with parameter freezing
const PAIRED_ITERS = 20000
history_paired = MVHistory()

println("\n" * "="^50)
println("Training Paired RBM")
println("="^50)

@time pcd!(
    rbm_paired, X_AB; 
    iters=PAIRED_ITERS, 
    batchsize=BATCH_SIZE, 
    steps=CD_STEPS,
    callback = function(; iter, _...)
        if iszero(iter % 100)
            lpl = mean(log_pseudolikelihood(rbm_paired, X_AB))
            println("iter: $iter lpl=$lpl")
            @trace history_paired iter lpl
        end
        # Re-freeze parameters after each update
        project_to_frozen_par!(rbm_paired, rbm_A, rbm_B, H_ADD)
    end
)

# ### Visualize Training Progress
fig = Makie.Figure(resolution=(500, 300))
ax = Makie.Axis(fig[1, 1], 
    xlabel="Iteration", 
    ylabel="Log-Pseudolikelihood",
    title="Paired RBM Training Progress")
Makie.lines!(ax, get(history_paired, :lpl)...)
fig


save_rbm("./rbmAB_k=30_tt=20k_dataset_20links_09strength.hdf5",rbm_paired)

# ## Free Energy Analysis for Paired RBM

fantasy_F_paired = zeros(N_SAMPLES, N_GIBBS_STEPS)
fantasy_x_paired = bitrand(N_VIS_TOTAL, N_SAMPLES)
fantasy_F_paired[:, 1] .= free_energy(rbm_paired, fantasy_x_paired)

println("\nRunning Gibbs sampling for paired RBM...")
@time for t in 2:N_GIBBS_STEPS
    fantasy_x_paired .= sample_v_from_v(rbm_paired, fantasy_x_paired, steps=100)
    fantasy_F_paired[:, t] .= free_energy(rbm_paired, fantasy_x_paired)
    println("step $t")
end

# ### Visualize Free Energy Convergence
fig = Makie.Figure(resolution=(400, 300))
ax = Makie.Axis(fig[1, 1], 
    xlabel="Sampling Step", 
    ylabel="Free Energy",
    title="Free Energy Convergence (Paired RBM)")

fantasy_F_μ_paired = vec(mean(fantasy_F_paired; dims=1))
fantasy_F_σ_paired = vec(std(fantasy_F_paired; dims=1))

Makie.band!(ax, 1:N_GIBBS_STEPS, 
    fantasy_F_μ_paired - fantasy_F_σ_paired/2, 
    fantasy_F_μ_paired + fantasy_F_σ_paired/2)
Makie.lines!(ax, 1:N_GIBBS_STEPS, fantasy_F_μ_paired)
fig

# ## Cross-Correlation Analysis

# ### Cross-correlations from paired RBM samples
C_AB_paired = empirical_cross_corr(
    fantasy_x_paired'[:, 1:30], 
    fantasy_x_paired'[:, 31:60])
C_AB_data = empirical_cross_corr(XA', XB')



# ### Cross-correlations from independent RBM samples
# Generate independent samples from individual RBMs
fantasy_x_A_ind = bitrand(N_VISIBLE, N_SAMPLES)
fantasy_x_B_ind = bitrand(N_VISIBLE, N_SAMPLES)

# Run Gibbs sampling independently
for t in 2:N_GIBBS_STEPS
    fantasy_x_A_ind .= sample_v_from_v(rbm_A, fantasy_x_A_ind, steps=50)
    fantasy_x_B_ind .= sample_v_from_v(rbm_B, fantasy_x_B_ind, steps=50)
end

C_AB_independent = empirical_cross_corr(fantasy_x_A_ind', fantasy_x_B_ind')

# Visualize cross-correlations
p_cross_model=heatmap(data.model.K, title="Model true correlations (A-B)")
p_cross = heatmap(C_AB_data, title="Data Cross-Correlations (A-B)")
p_cross_paired = heatmap(C_AB_paired, title="Paired RBM Cross-Correlations (A-B)")
p_cross_indep = heatmap(C_AB_independent, title="Independent RBMs Cross-Correlations (A-B)")

fig=Plots.plot(p_cross_model,p_cross, p_cross_paired, p_cross_indep, layout=(1, 4), size=(2500, 300))
savefig("./corr_coparison_dataset_20links_09strength_singlett=30k_pairedtt=20k_k=30.png")

# ## Validation Tests for Paired RBM

# ### Paired RBM: Hidden Unit Activations
println("\n" * "="^50)
println("Validation: Paired RBM Hidden Activations")
println("="^50)

h_data_paired = sample_h_from_v(rbm_paired, X_AB) .|> Float64
h_model_paired = sample_h_from_v(rbm_paired, fantasy_x_paired) .|> Float64

h_data_mean_paired = vec(mean(h_data_paired; dims=2))
h_model_mean_paired = vec(mean(h_model_paired; dims=2))

p7 = plot_activation_comparison(h_data_mean_paired, h_model_mean_paired, 
    "Hidden Activations - Paired RBM", "⟨h⟩ data", "⟨h⟩ model")

# ### Paired RBM: Visible Unit Activations
println("\n" * "="^50)
println("Validation: Paired RBM Visible Activations")
println("="^50)

v_data_mean_paired = vec(mean(X_AB; dims=2))
v_model_mean_paired = vec(mean(fantasy_x_paired; dims=2))

p8 = plot_activation_comparison(v_data_mean_paired, v_model_mean_paired, 
    "Visible Activations - Paired RBM", "⟨v⟩ data", "⟨v⟩ model")

# ### Paired RBM: H-V Correlations
println("\n" * "="^50)
println("Validation: Paired RBM H-V Correlations")
println("="^50)

hv_data_paired = h_data_paired * X_AB' / size(X_AB, 2)
hv_model_paired = h_model_paired * fantasy_x_paired' / size(fantasy_x_paired, 2)

hv_data_flat_paired = vec(hv_data_paired)
hv_model_flat_paired = vec(hv_model_paired)

p9 = plot_activation_comparison(hv_data_flat_paired, hv_model_flat_paired, 
    "H-V Correlations - Paired RBM", "⟨hv⟩ data", "⟨hv⟩ model")
# ## Family Statistics Preservation Tests
#
# Test whether the paired RBM preserves individual family statistics
# by splitting paired samples into their constituent families.

# Split paired samples into A and B components
fantasy_x_paired_A = fantasy_x_paired[1:30, :]
fantasy_x_paired_B = fantasy_x_paired[31:60, :]

# ### Test 1: Hidden activation preservation for family A
println("\n" * "="^50)
println("Family Statistics Preservation - Family A")
println("="^50)

h_data_A_from_paired = sample_h_from_v(rbm_A, XA) .|> Float64
h_model_A_from_paired = sample_h_from_v(rbm_A, fantasy_x_paired_A) .|> Float64

h_data_mean_A_paired = vec(mean(h_data_A_from_paired; dims=2))
h_model_mean_A_paired = vec(mean(h_model_A_from_paired; dims=2))

p10 = plot_activation_comparison(h_data_mean_A_paired, h_model_mean_A_paired, 
    "Hidden Activations - Family A (from paired RBM)", "⟨h⟩ data", "⟨h⟩ model")

# Visible activations
v_data_mean_A_from_paired = vec(mean(XA; dims=2))
v_model_mean_A_from_paired = vec(mean(fantasy_x_paired_A; dims=2))

p11 = plot_activation_comparison(v_data_mean_A_from_paired, v_model_mean_A_from_paired, 
    "Visible Activations - Family A (from paired RBM)", "⟨v⟩ data", "⟨v⟩ model")

# H-V correlations
hv_data_A_paired = h_data_A_from_paired * XA' / size(XA, 2)
hv_model_A_paired = h_model_A_from_paired * fantasy_x_paired_A' / size(fantasy_x_paired_A, 2)

hv_data_flat_A_paired = vec(hv_data_A_paired)
hv_model_flat_A_paired = vec(hv_model_A_paired)

p12 = plot_activation_comparison(hv_data_flat_A_paired, hv_model_flat_A_paired, 
    "H-V Correlations - Family A (from paired RBM)", "⟨hv⟩ data", "⟨hv⟩ model")

Plots.plot(p10, p11, p12, layout=(1, 3), size=(1200, 400))

# ### Test 2: Hidden activation preservation for family B
println("\n" * "="^50)
println("Family Statistics Preservation - Family B")
println("="^50)

h_data_B_from_paired = sample_h_from_v(rbm_B, XB) .|> Float64
h_model_B_from_paired = sample_h_from_v(rbm_B, fantasy_x_paired_B) .|> Float64

h_data_mean_B_paired = vec(mean(h_data_B_from_paired; dims=2))
h_model_mean_B_paired = vec(mean(h_model_B_from_paired; dims=2))

p13 = plot_activation_comparison(h_data_mean_B_paired, h_model_mean_B_paired, 
    "Hidden Activations - Family B (from paired RBM)", "⟨h⟩ data", "⟨h⟩ model")

# Visible activations
v_data_mean_B_from_paired = vec(mean(XB; dims=2))
v_model_mean_B_from_paired = vec(mean(fantasy_x_paired_B; dims=2))

p14 = plot_activation_comparison(v_data_mean_B_from_paired, v_model_mean_B_from_paired, 
    "Visible Activations - Family B (from paired RBM)", "⟨v⟩ data", "⟨v⟩ model")

# H-V correlations
hv_data_B_paired = h_data_B_from_paired * XB' / size(XB, 2)
hv_model_B_paired = h_model_B_from_paired * fantasy_x_paired_B' / size(fantasy_x_paired_B, 2)

hv_data_flat_B_paired = vec(hv_data_B_paired)
hv_model_flat_B_paired = vec(hv_model_B_paired)

p15 = plot_activation_comparison(hv_data_flat_B_paired, hv_model_flat_B_paired, 
    "H-V Correlations - Family B (from paired RBM)", "⟨hv⟩ data", "⟨hv⟩ model")

Plots.plot(p13, p14, p15, layout=(1, 3), size=(1200, 400))

# ### Family correlation preservation
println("\n" * "="^50)
println("Family Correlation Preservation")
println("="^50)

# Individual family correlations from paired samples
CA_paired_model = empirical_corr(fantasy_x_paired_A')
CB_paired_model = empirical_corr(fantasy_x_paired_B')

# Compare with original data correlations
p_corr_A = heatmap(CA_data, title="Data Correlations (Family A)")
p_corr_A_paired = heatmap(CA_paired_model, title="Paired RBM Correlations (Family A)")
p_corr_B = heatmap(CB_data, title="Data Correlations (Family B)")
p_corr_B_paired = heatmap(CB_paired_model, title="Paired RBM Correlations (Family B)")

Plots.plot(p_corr_A, p_corr_A_paired, p_corr_B, p_corr_B_paired, layout=(2, 2), size=(800, 800))

# Print correlation comparison
println("\nCorrelation preservation (off-diagonal mean):")
println("Family A - Data: $(round(mean(abs.(CA_data - diagm(diag(CA_data)))), digits=6))")
println("Family A - Paired RBM: $(round(mean(abs.(CA_paired_model - diagm(diag(CA_paired_model)))), digits=6))")
println("Family B - Data: $(round(mean(abs.(CB_data - diagm(diag(CB_data)))), digits=6))")
println("Family B - Paired RBM: $(round(mean(abs.(CB_paired_model - diagm(diag(CB_paired_model)))), digits=6))")

# ## Save Paired RBM

const PAIRED_RBM_PATH = "./rbmAB_hadd=$(H_ADD)_k=30_tt=$(PAIRED_ITERS)_example_dataset.hdf5"
save_rbm(PAIRED_RBM_PATH, rbm_paired)
println("\nSaved paired RBM to $PAIRED_RBM_PATH")

# ## Summary
#
# This analysis demonstrated:
# 1. Training individual RBMs on both coupled spin systems (A and B)
# 2. Monitoring convergence through log-pseudolikelihood
# 3. Validating learned distributions via free energy analysis
# 4. Comprehensive model validation through activation comparisons:
#    - Hidden unit activations (⟨h⟩ data vs ⟨h⟩ model)
#    - Visible unit activations (⟨v⟩ data vs ⟨v⟩ model)
#    - Hidden-visible correlations (⟨hv⟩ data vs ⟨hv⟩ model)
# 5. Combining RBMs into a paired architecture with frozen base parameters
# 6. Training additional hidden units to capture cross-system interactions
# 7. Cross-correlation analysis showing paired RBM captures inter-family correlations
#    while independent RBMs do not
# 8. Validation that paired RBM preserves individual family statistics
# 9. Family-specific correlation preservation analysis

println("\n" * "="^50)
println("Analysis Complete!")
println("="^50)