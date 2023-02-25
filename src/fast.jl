
function cor_symmetrical(X, Y=nothing; avoid_copy=false)
    X, Y = cor_preprocess_symmetrical(X, Y, avoid_copy=avoid_copy)
    return X'*Y
end


function cor_asymmetrical(X, Y=nothing)
    X, Y = cor_preprocess_asymmetrical(X, Y)
    return X'*Y
end


function col_normout(A)
    # col-normalize a matrix
   ES = enumerate(vec(sum(A,dims=1)))
   B = zeros(size(A,1),size(A,2))
   for (col,s) in ES
       s==0 && continue
       B[:,col] = A[:,col]./s
   end
   return B
end
