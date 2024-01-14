import Base.sqrt

"""
     rank(A::AbstractTensorMap) -> ::Int64
     
Return the rank of a given tensor.

     rank(A::AbstractTensorMap, idx::Int64) -> ::Int64
Return the rank corresponding to codomain (`idx = 1`) or domain (`idx = 2`). 
"""
function rank(A::AbstractTensorMap, idx::Int64)
     @assert idx ∈ [1, 2]
     idx == 1 && return typeof(codomain(A)).parameters[2]
     idx == 2 && return typeof(domain(A)).parameters[2]
end
rank(A::AbstractTensorMap) = rank(A, 1) + rank(A, 2)


"""
Diagnal tensorMap `S` -> `√S^-1`
"""
function inv_sqrt!(S::TensorMap)
     for (c, blk) in blocks(S)
          for ii in 1:size(blk)[1]
               block(S, c)[ii, ii] = sqrt(one(block(S, c)[ii, ii]) / block(S, c)[ii, ii])
          end
     end
     return S
end

"""
Diagnal tensorMap `S` -> `√S^-1`.
Also normalize at the same time.
"""
function inv_sqrt(S::TensorMap)
     Sid = deepcopy(S)
     normalize!(Sid)
     for (c, blk) in blocks(Sid)
          for ii in 1:size(blk)[1]
               block(Sid, c)[ii, ii] = sqrt(one(block(S, c)[ii, ii]) / block(S, c)[ii, ii])
          end
     end
     return Sid
end

"""
Diagnal tensorMap `S` -> `√S^`.
"""
function sqrt(S::TensorKit.TensorMap)
     Sid = deepcopy(S)
     for (c, blk) in blocks(Sid)
          for ii in 1:size(blk)[1]
               block(Sid, c)[ii, ii] = sqrt(block(S, c)[ii, ii])
          end
     end
     return Sid
end

"""
Diagnal tensorMap `S` -> `S^-1`.
"""
function inv(S::TensorMap)
     Sid = deepcopy(S)
     for (c, blk) in blocks(Sid)
          for ii in 1:size(blk)[1]
               block(Sid, c)[ii, ii] = one(block(S, c)[ii, ii]) / block(S, c)[ii, ii]
          end
     end
     return Sid
end