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
get the bra state iPEPS tensor. Nothing but M† with two swap gates contracted.
"""
function bar(M::TensorMap)
     # get ̄M for fermionic iPEPS tensor. ̄M is nothing but M† with two swap gates.
     gatel = swap_gate(space(M)[1], space(M)[2])  # left and top
     gater = swap_gate(space(M)[4]', space(M)[5]')  # right and bottom
     # 这一行的 in 后缀是内部指标，不代表方向
     @tensor Mbar[l, t, p; r, b] := gatel[lin, tin, l, t] * M'[rin, bin, lin, tin, p] * gater[r, b, rin, bin]
     return Mbar
end

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