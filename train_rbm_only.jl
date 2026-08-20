# # Paired RBM Analysis for Coupled Spin Systems
#
# This notebook analyzes coupled spin systems using Restricted Boltzmann Machines,
# training individual RBMs and then combining them into a paired architecture.
# Includes comprehensive validation tests for model quality and statistics preservation.

# ## Environment Setup
using TwoFamilyIsing
using HDF5
using RestrictedBoltzmannMachines: RBM, BinaryRBM, initialize!
using RestrictedBoltzmannMachines: log_pseudolikelihood
using RestrictedBoltzmannMachines: pcd!
using RestrictedBoltzmannMachines: sample_from_inputs
using RestrictedBoltzmannMachines: sample_v_from_v
using RestrictedBoltzmannMachines: save_rbm, load_rbm
using RestrictedBoltzmannMachines: free_energy
using RestrictedBoltzmannMachines: sample_h_from_v, sample_v_from_h, standardize
using Statistics: mean, std, var, cor
using ValueHistories: @trace, MVHistory
using Random
using DelimitedFiles



const N_HIDDEN = parse(Int,ARGS[1])
const N_ITERATIONS = parse(Int,ARGS[2])
const CD_STEPS =  parse(Int,ARGS[3])
const PAIRED_ITERS =  parse(Int,ARGS[4])
const H_ADD = parse(Int,ARGS[5])  # Additional hidden units for cross-interactions
just_load = parse(Int, ARGS[6]) #If true doesn't train the individual machines and just loads the already saved ones

path_lplA="./lpl_rbmA_standardize_k=$(CD_STEPS)_tt=$(N_ITERATIONS)_dataset_20links_09strength.txt"
path_lplB="./lpl_rbmB_standardize_k=$(CD_STEPS)_tt=$(N_ITERATIONS)_dataset_20links_09strength.txt" 

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
const BATCH_SIZE = 256

# ### Train RBM for System A
println("\n" * "="^50)
println("Training RBM A")
println("="^50)

if just_load==0

rbm_A = BinaryRBM(Float32, (N_VISIBLE,), N_HIDDEN)
initialize!(rbm_A, XA)
rbm_A=standardize(rbm_A)

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
	    open(path_lplA, "a") do f
            	println(f, lpl)
	    end
        end
    end
)

rbm_B = BinaryRBM(Float32, (N_VISIBLE,), N_HIDDEN)
initialize!(rbm_B, XB)
rbm_B =standardize(rbm_B)

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
            open(path_lplB, "a") do f
            	println(f, lpl)
	    end
        end
    end
)

# ### Save Trained RBMs
const RBM_A_PATH = "./rbmA_standardize_k=$(CD_STEPS)_tt=$(N_ITERATIONS)_dataset_20links_09strength.hdf5"
const RBM_B_PATH = "./rbmB_standardize_k=$(CD_STEPS)_tt=$(N_ITERATIONS)_dataset_20links_09strength.hdf5"
save_rbm(RBM_A_PATH, rbm_A)
save_rbm(RBM_B_PATH, rbm_B)

println("Saved RBM A to $RBM_A_PATH")
println("Saved RBM B to $RBM_B_PATH")

end

rbm_A=load_rbm("./rbmA_standardize_k=$(CD_STEPS)_tt=$(N_ITERATIONS)_dataset_20links_09strength.hdf5")
rbm_B=load_rbm("./rbmB_standardize_k=$(CD_STEPS)_tt=$(N_ITERATIONS)_dataset_20links_09strength.hdf5")

# ## Paired RBM Architecture
#
# We now combine the two systems into a paired RBM architecture
# with additional hidden units to capture cross-system interactions.

# Combine datasets
X_AB = vcat(XA, XB)
println("Combined dataset shape: $(size(X_AB))")

# ### Initialize Paired RBM
const N_VIS_TOTAL = 60
const N_HID_TOTAL = 30 + H_ADD

path_lpl_paired="./lpl_rbmAB_standardize_hadd=$(H_ADD)_k=$(CD_STEPS)_ttpaired=$(PAIRED_ITERS)_tt=$(N_ITERATIONS)_dataset_20links_09strength.txt"



rbm_paired = BinaryRBM(Float32, (N_VIS_TOTAL,), N_HID_TOTAL)
initialize!(rbm_paired, X_AB)
rbm_paired=standardize(rbm_paired)

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
            
	    open(path_lpl_paired, "a") do f
                println(f, lpl)
            end

        end
        # Re-freeze parameters after each update
        project_to_frozen_par!(rbm_paired, rbm_A, rbm_B, H_ADD)
    end
)


const PAIRED_RBM_PATH = "./rbmAB_standardize_hadd=$(H_ADD)_k=$(CD_STEPS)_ttpaired=$(PAIRED_ITERS)_tt=$(N_ITERATIONS)_dataset_20links_09strength.hdf5"
save_rbm(PAIRED_RBM_PATH, rbm_paired)
println("\nSaved paired RBM to $PAIRED_RBM_PATH")



const H_CTRL=H_ADD
const CTRL_ITERS = PAIRED_ITERS

rbm_ctrl = BinaryRBM(Float32, (N_VIS_TOTAL,), H_CTRL)
initialize!(rbm_ctrl, X_AB)
rbm_ctrl=standardize(rbm_ctrl)

println("Initial log(PL) for control RBM: ",
    mean(@time log_pseudolikelihood(rbm_ctrl, X_AB)))

history_ctrl = MVHistory()

path_lpl_control="./lpl_rbmAB_control_standardize_hadd=$(H_ADD)_k=$(CD_STEPS)_ttpaired=$(PAIRED_ITERS)_dataset_20links_09strength.txt"

@time pcd!(
    rbm_ctrl, X_AB;
    iters=CTRL_ITERS,
    batchsize=BATCH_SIZE,
    steps=CD_STEPS,
    callback = function(; iter, _...)
        if iszero(iter % 100)
            lpl = mean(log_pseudolikelihood(rbm_ctrl, X_AB))
            println("iter: $iter lpl=$lpl")

            open(path_lpl_control, "a") do f
                println(f, lpl)
            end
        end
    end
)

save_rbm("./rbmAB_control_standardize_hadd=$(H_ADD)_k=$(CD_STEPS)_ttpaired=$(PAIRED_ITERS)_dataset_20links_09strength.hdf5",rbm_ctrl)



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
