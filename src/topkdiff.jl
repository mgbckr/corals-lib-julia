function topkdiff_matrix(X1, X2, Y1=nothing, Y2=nothing; k=nothing, fsort=false)
    
    # println("correlation")
    diff = cor_symmetrical(X1, Y1, avoid_copy=true) - cor_symmetrical(X2, Y2, avoid_copy=true)
    
    if k === nothing
        k = trunc(Int, length(diff) * 0.01)  # top 1%
    end
    k = min(k, length(diff))
    # println(k)

    # println("sorting")
    if fsort
        idx = fsortperm(vec(-abs.(diff)))[1:k]
    else
        idx = sortperm(vec(abs.(diff)), rev=true)[1:k]
    end
    
    # print("return")
    return diff[idx], Tuple.(CartesianIndices(size(diff))[idx])

end

