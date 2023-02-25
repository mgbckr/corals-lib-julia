using Statistics
using Printf


function cor_preprocess_asymmetrical(X, Y=nothing)

    X = X ./ sum(X, dims=1)
    if Y === nothing
        Y = X  # TODO: copy as in Python?
    else
        Y = Y ./ sum(Y, dims=1)
    end

    n = size(X)[1]
    X = (X .- 1/n) ./ sqrt.(sum((X .- 1/n).^2, dims=1))
    Y = Y ./ sqrt.(sum((Y .- 1/n).^2, dims=1))
    
    return X, Y
end

function cor_preprocess_symmetrical(X, Y=nothing; avoid_copy=false)

    X = cor_preprocess(X)

    if Y === nothing
        if avoid_copy
            Y = X  # TODO: copy as in Python?
        else
            Y = copy(X)
        end
    else
        Y = cor_preprocess(Y)
    end

    return X, Y
end


function cor_preprocess(X)
    # TODO: can be slightly optimized by reusing (x - mu) and dropping sqrt(m)
    X = X .- mean(X, dims=1)
    X ./= (std(X, corrected=false, dims=1) .* sqrt(size(X, 1)))
    return X
end


function derive_bins(n, n_batches)
    bin_size_default = div(n, n_batches)

    bin_sizes = repeat([bin_size_default], n_batches)
    for i = 1:(n - bin_size_default * n_batches)
        bin_sizes[mod(i, length(bin_sizes))] += 1
    end

    # 1 indexing is totally weird
    bins = cumsum(bin_sizes)
    bins[end] += 1
    pushfirst!(bins, 1)
    
    return bins
end
