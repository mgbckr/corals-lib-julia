using Test
include("../src/topk.jl")


@testset "topk" begin

    test_top8_data = [
        1 1 -1 1;
        2 1 -2 2;
        3 3 -3 1;
        4 4 -3 1.;
    ]

    test_top8_cor = [1,1,1,1,0.94672926,0.94672926,-0.94387981,-0.94387981]

    test_top8_idx= [
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 1),
        (2, 2),
        (3, 1),
        (3, 3),
        (4, 4)
    ]

    val, idx = topk_matrix(test_top8_data, k=8)
    @test isapprox(val, test_top8_cor)
    @test sort(idx) == test_top8_idx

    val, idx = topk_balltree_mlpack_twice(test_top8_data, k=8)
    @test isapprox(val, test_top8_cor)
    @test sort(idx) == test_top8_idx

    val, idx = topk_balltree_mlpack_combined_tree(test_top8_data, k=8)
    @test isapprox(val, test_top8_cor)
    @test sort(idx) == test_top8_idx

    val, idx = topk_balltree_nn_combined_tree(test_top8_data, k=8)
    @test isapprox(val, test_top8_cor)
    @test sort(idx) == test_top8_idx

    val, idx = topk_balltree_nn_combined_tree_parallel(test_top8_data, k=8)
    @test isapprox(val, test_top8_cor)
    @test sort(idx) == test_top8_idx

    val, idx = topk_balltree_mlpack_combined_query(test_top8_data, k=8)
    @test isapprox(val, test_top8_cor)
    @test sort(idx) == test_top8_idx

    val, idx = topk_balltree_mlpack_combined_query_parallel(test_top8_data, k=8, n_batches=3)
    @test isapprox(val, test_top8_cor)
    @test sort(idx) == test_top8_idx

    val, idx = topk_balltree_nn_combined_query(test_top8_data, k=8)
    @test isapprox(val, test_top8_cor)
    @test sort(idx) == test_top8_idx

    val, idx = topk_balltree_nn_combined_query_parallel(test_top8_data, k=8)
    @test isapprox(val, test_top8_cor)
    @test sort(idx) == test_top8_idx

end;