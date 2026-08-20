struct FamilySpec
    n::Int
    structure::Symbol              # :modular or :sparse_random

    # modular options
    module_sizes::Vector{Int}
    p_in::Float64
    p_out::Float64
    J_in_range::Tuple{Float64,Float64}
    J_out_range::Tuple{Float64,Float64}

    # sparse random options
    edge_density::Float64
    J_range::Tuple{Float64,Float64}
    allow_negative::Bool

    # common
    field_range::Tuple{Float64,Float64}
end

struct CrossSpec
    structure::Symbol              # :sparse_pairs or :sparse_block
    n_links::Int
    K_range::Tuple{Float64,Float64}
    allow_negative::Bool

    # optional block coupling
    a_modules::Vector{Int}
    b_modules::Vector{Int}
end

struct ModelSpec
    familyA::FamilySpec
    familyB::FamilySpec
    cross::CrossSpec
    beta::Float64
    seed::Int
end

struct IsingModel
    JA::Matrix{Float64}
    JB::Matrix{Float64}
    K::Matrix{Float64}
    hA::Vector{Float64}
    hB::Vector{Float64}
    beta::Float64

    modulesA::Vector{Int}
    modulesB::Vector{Int}

    cross_edges::Vector{Tuple{Int,Int,Float64}}
end

struct IsingDataset
    spec::ModelSpec
    model::IsingModel

    X_joint_train::Matrix{Int8}
    X_joint_val::Matrix{Int8}
    X_joint_test::Matrix{Int8}

    XA_train::Matrix{Int8}
    XA_val::Matrix{Int8}
    XA_test::Matrix{Int8}

    XB_train::Matrix{Int8}
    XB_val::Matrix{Int8}
    XB_test::Matrix{Int8}

    metadata::Dict{Symbol,Any}
end
