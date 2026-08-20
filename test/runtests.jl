using Test
using TwoFamilyIsing

@testset "TwoFamilyIsing basic build and sample" begin
    familyA = FamilySpec(
        12, :modular,
        [6, 6],
        0.8, 0.1,
        (0.4, 0.8),
        (0.0, 0.1),
        0.0,
        (0.0, 0.0),
        false,
        (-0.01, 0.01)
    )

    familyB = FamilySpec(
        10, :sparse_random,
        Int[],
        0.0, 0.0,
        (0.0, 0.0),
        (0.0, 0.0),
        0.2,
        (-0.6, 0.6),
        true,
        (-0.01, 0.01)
    )

    cross = CrossSpec(
        :sparse_pairs,
        5,
        (-0.5, 0.5),
        true,
        Int[],
        Int[]
    )

    spec = ModelSpec(familyA, familyB, cross, 1.0, 42)
    model = build_model(spec)

    @test size(model.JA) == (12, 12)
    @test size(model.JB) == (10, 10)
    @test size(model.K) == (12, 10)
    @test length(model.cross_edges) == 5

    X = sample_joint(model; n_samples=100, burnin=500, thinning=5, seed=7)
    @test size(X) == (100, 22)
    @test all(x -> x == -1 || x == 1, X)

    data = make_dataset(spec; n_total_samples=200, burnin=500, thinning=5)
    @test size(data.X_joint_train, 2) == 22
    @test size(data.XA_train, 2) == 12
    @test size(data.XB_train, 2) == 10
end
