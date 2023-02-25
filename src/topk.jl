import NearestNeighbors
# using SortingLab
using mlpack: knn


function topk_matrix(X, Y=nothing; k=nothing, fsort=false)
    
    # println("correlation")
    cor = cor_symmetrical(X, Y, avoid_copy=true)
    
    if k === nothing
        k = trunc(Int, length(cor) * 0.01)  # top 1%
    end
    k = min(k, length(cor))
    # println(k)

    # println("sorting")
    if fsort
        idx = fsortperm(vec(-abs.(cor)))[1:k]
    else
        idx = sortperm(vec(abs.(cor)), rev=true)[1:k]
    end
    
    # print("return")
    return cor[idx], Tuple.(CartesianIndices(size(cor))[idx])

end


function topk_balltree_mlpack_twice(X, Y=nothing; k=nothing, approximation_factor=10)

    X, Y = cor_preprocess_symmetrical(X, Y)
    k = derive_k(k, X, Y)
    kk = derive_k_per_query(k, X, Y, approximation_factor)

    dst1, idx1, model = knn(reference=X', query=Y', k=kk, tree_type="ball", algorithm="dual_tree", leaf_size=40)
    dst2, idx2, _ = knn(query=-Y', k=kk, input_model=model)

    idx1_r = reduce(vcat, map(x -> repeat([x[1]], length(x[2])), enumerate(eachrow(idx1))))
    idx1_c = vec(idx1')

    idx2_r = reduce(vcat, map(x -> repeat([x[1]], length(x[2])), enumerate(eachrow(idx2))))
    idx2_c = vec(idx2')
    # idx2_c = reduce(vcat, idx2)  # eats memory like crazy

    idx_r = vcat(idx1_r, idx2_r)
    idx_c = vcat(idx1_c, idx2_c)

    mask_inverse = (length(idx1_r) + 1):(length(idx1_r) + length(idx2_r))

    return derive_topk(mask_inverse, vcat(dst1, dst2), idx_r, idx_c, k)
end


function topk_balltree_mlpack_combined_query(X, Y=nothing; k=nothing, approximation_factor=10)

    X, Y = cor_preprocess_symmetrical(X, Y)
    k = derive_k(k, X, Y)
    kk = derive_k_per_query(k, X, Y, approximation_factor)

    query = hcat(Y, -Y)

    dst, idx, model = knn(reference=X', query=query', k=kk, tree_type="ball", algorithm="dual_tree", leaf_size=40)

    idx_r = reduce(vcat, map(x -> repeat([x[1]], length(x[2])), enumerate(eachrow(idx))))
    idx_c = vec(idx')

    # get mask for selecting correlations from -Y
    mask_inverse = findall(idx_r .> size(Y, 2))

    # fix index for correlations from -Y
    idx_r[mask_inverse] .-= size(Y, 2)

    return derive_topk(mask_inverse, dst, idx_r, idx_c, k)
end


function topk_balltree_mlpack_combined_tree(X, Y=nothing; k=nothing, approximation_factor=10)

    X, Y = cor_preprocess_symmetrical(X, Y)
    k = derive_k(k, X, Y)
    kk = derive_k_per_query(k, X, Y, approximation_factor)

    ref = hcat(X, -X)

    dst, idx, _ = knn(reference=ref', query=Y', k=kk, tree_type="ball", algorithm="dual_tree", leaf_size=40)

    idx_r = reduce(vcat, map(x -> repeat([x[1]], length(x[2])), enumerate(eachrow(idx))))
    idx_c = vec(idx')

    # get mask for selecting correlations from -Y
    mask_inverse = findall(idx_c .> size(X, 2))

    # fix index for correlations from -Y
    idx_c[mask_inverse] .-= size(Y, 2)

    return derive_topk(mask_inverse, dst, idx_r, idx_c, k)
end


"""
WARNING: This guy doesn't work at all when `Threads.nthreads() > 0`. 
It seems mlpack does not play well with threading. 
"""
function topk_balltree_mlpack_combined_query_parallel(
        X, Y=nothing; 
        k=nothing, approximation_factor=10, 
        n_batches=nothing)

    if n_batches === nothing
        n_batches = Base.Threads.nthreads()
    end

    X, Y = cor_preprocess_symmetrical(X, Y)
    k = derive_k(k, X, Y)
    kk = derive_k_per_query(k, X, Y, approximation_factor)

    n_batches = min(size(Y, 2), n_batches)

    query = hcat(Y, -Y)

    # TODO: this is a bit weird since I kind of cheat to build the model; for k=0 the process just crashes
    _, _, model = knn(reference=X', query=X[:,1]', k=1, tree_type="ball", algorithm="dual_tree", leaf_size=40)
    
    bins = derive_bins(size(query, 2), n_batches)

    dst = Vector{Array}(undef, n_batches)
    idx = Vector{Array}(undef, n_batches)
    Threads.@threads for i in 1:n_batches
        # 1 indexing is totally weird since `derive_bins` needs to do weird magic
        batch_query = query[:,bins[i]:bins[i+1] - 1]
        batch_dst, batch_idx, _ = knn(input_model=model, query=batch_query', k=kk)
        dst[i] = batch_dst
        idx[i] = batch_idx
    end

    dst = reduce(vcat, dst)
    idx = reduce(vcat, idx)

    idx_r = reduce(vcat, map(x -> repeat([x[1]], length(x[2])), enumerate(eachrow(idx))))
    idx_c = vec(idx')

    # get mask for selecting correlations from -Y
    mask_inverse = findall(idx_r .> size(Y, 2))
    
    # fix index for correlations from -Y
    idx_r[mask_inverse] .-= size(Y, 2)

    return derive_topk(mask_inverse, dst, idx_r, idx_c, k)

end


function topk_balltree_nn_combined_tree(X, Y=nothing; k=nothing, approximation_factor=10)

    X, Y = cor_preprocess_symmetrical(X, Y)
    k = derive_k(k, X, Y)
    kk = derive_k_per_query(k, X, Y, approximation_factor)

    ref = hcat(X, -X)

    tree = NearestNeighbors.BallTree(ref)
    idx, dst = NearestNeighbors.knn(tree, Y, kk)

    idx_r = reduce(vcat, map(x -> repeat([x[1]], length(x[2])), enumerate(idx)))
    idx_c = reduce(vcat, idx)

    # get mask for selecting correlations from -Y
    mask_inverse = findall(idx_c .> size(X, 2))

    # fix index for correlations from -Y
    idx_c[mask_inverse] .-= size(Y, 2)

    return derive_topk(mask_inverse, reduce(vcat, dst), idx_r, idx_c, k)
end


function topk_balltree_nn_combined_tree_parallel(
        X, Y=nothing; k=nothing, approximation_factor=10,
        n_batches=nothing)

    if n_batches === nothing
        n_batches = Base.Threads.nthreads()
    end

    X, Y = cor_preprocess_symmetrical(X, Y)
    k = derive_k(k, X, Y)
    kk = derive_k_per_query(k, X, Y, approximation_factor)

    n_batches = min(size(Y, 2), n_batches)
    
    ref = hcat(X, -X)

    tree = NearestNeighbors.BallTree(ref)
    
    bins = derive_bins(size(Y, 2), n_batches)

    dst = Vector{Array}(undef, n_batches)
    idx = Vector{Array}(undef, n_batches)
    Threads.@threads for i in 1:n_batches

        # 1 indexing is totally weird since `derive_bins` needs to do weird magic
        batch_query = Y[:, bins[i]:bins[i+1] - 1]
        
        batch_idx, batch_dst = NearestNeighbors.knn(tree, batch_query, kk)
        
        dst[i] = batch_dst
        idx[i] = batch_idx
    end
    dst = reduce(vcat, dst)
    idx = reduce(vcat, idx)

    idx_r = reduce(vcat, map(x -> repeat([x[1]], length(x[2])), enumerate(idx)))
    idx_c = reduce(vcat, idx)

    # get mask for selecting correlations from -Y
    mask_inverse = findall(idx_c .> size(X, 2))

    # fix index for correlations from -Y
    idx_c[mask_inverse] .-= size(Y, 2)

    return derive_topk(mask_inverse, reduce(vcat, dst), idx_r, idx_c, k)
end


function topk_balltree_nn_combined_query(X, Y=nothing; k=nothing, approximation_factor=10)

    X, Y = cor_preprocess_symmetrical(X, Y)
    k = derive_k(k, X, Y)
    kk = derive_k_per_query(k, X, Y, approximation_factor)

    query = hcat(Y, -Y)

    tree = NearestNeighbors.BallTree(X)
    idx, dst = NearestNeighbors.knn(tree, query, kk)

    idx_r = reduce(vcat, map(x -> repeat([x[1]], length(x[2])), enumerate(idx)))
    idx_c = reduce(vcat, idx)

    # get mask for selecting correlations from -Y
    mask_inverse = findall(idx_r .> size(Y, 2))

    # fix index for correlations from -Y
    idx_r[mask_inverse] .-= size(Y, 2)
    
    return derive_topk(mask_inverse, reduce(vcat, dst), idx_r, idx_c, k)

end


function topk_balltree_nn_combined_query_parallel(
        X, Y=nothing; 
        k=nothing, approximation_factor=10, 
        n_batches=nothing)

    if n_batches === nothing
        n_batches = Base.Threads.nthreads()
    end

    X, Y = cor_preprocess_symmetrical(X, Y)
    k = derive_k(k, X, Y)
    kk = derive_k_per_query(k, X, Y, approximation_factor)

    n_batches = min(size(Y, 2), n_batches)
    
    query = hcat(Y, -Y)
    
    tree = NearestNeighbors.BallTree(X)
    
    bins = derive_bins(size(query, 2), n_batches)

    dst = Vector{Array}(undef, n_batches)
    idx = Vector{Array}(undef, n_batches)
    Threads.@threads for i in 1:n_batches

        # 1 indexing is totally weird since `derive_bins` needs to do weird magic
        batch_query = query[:,bins[i]:bins[i+1] - 1]
        
        batch_idx, batch_dst = NearestNeighbors.knn(tree, batch_query, kk)
        
        dst[i] = batch_dst
        idx[i] = batch_idx
    end
    dst = reduce(vcat, dst)
    idx = reduce(vcat, idx)

    idx_r = reduce(vcat, map(x -> repeat([x[1]], length(x[2])), enumerate(idx)))
    idx_c = reduce(vcat, idx)

    # get mask for selecting correlations from -Y
    mask_inverse = findall(idx_r .> size(Y, 2))

    # fix index for correlations from -Y
    idx_r[mask_inverse] .-= size(Y, 2)    

    return derive_topk(mask_inverse, reduce(vcat, dst), idx_r, idx_c, k)

end

function derive_k(k, X, Y)
    if k === nothing
        k = trunc(Int, size(X, 2) * size(Y, 2) * 0.01)  # top 10%
    end
    k = min(k, size(X, 2) * size(Y, 2))
    
    return k
end


function derive_k_per_query(k, X, Y, approximation_factor)
    return min(
        trunc(Int, ceil(k / size(Y, 2))) * approximation_factor, 
        size(X, 2))
end


function derive_topk(mask_inverse, distances, idx_r, idx_c, k)

    correlation_values = vec(1 .- distances'.^2 ./2)
    order = sortperm(correlation_values, rev=true)[1:k]

    # fix correlations from -Yh
    if mask_inverse !== nothing
        correlation_values[mask_inverse] .*= -1
    end
    
    # correlation
    return correlation_values[order], Tuple.(zip(idx_r[order], idx_c[order]))

end