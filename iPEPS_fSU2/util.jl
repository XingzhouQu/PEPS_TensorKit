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
rank(A::AbstractTensorMap) = numind(A::AbstractTensorMap)


"""
Diagnal tensorMap `S` -> `√S^-1`
"""
function inv_sqrt!(S::TensorMap)
     # for (c, blk) in blocks(S)
     #      for ii in 1:size(blk)[1]
     #           block(S, c)[ii, ii] = sqrt(one(block(S, c)[ii, ii]) / block(S, c)[ii, ii])
     #      end
     # end
     S = inv(sqrt(S))
     return S
end

"""
Diagnal tensorMap `S` -> `√(S^-1)`.
"""
function inv_sqrt(S::TensorMap)
     Sid = sqrt(inv(S))
     # for (c, blk) in blocks(Sid)
     #      for ii in 1:size(blk)[1]
     #           block(Sid, c)[ii, ii] = sqrt(one(block(S, c)[ii, ii]) / block(S, c)[ii, ii])
     #      end
     # end
     return Sid
end

# """
#     sqrt(S::TensorKit.TensorMap)

# Diagnal tensorMap `S` -> `√S`.
# """
# function Base.sqrt(S::TensorKit.TensorMap)
#      Sid = deepcopy(S)
#      for (c, blk) in blocks(Sid)
#           for ii in 1:size(blk)[1]
#                block(Sid, c)[ii, ii] = sqrt(block(S, c)[ii, ii])
#           end
#      end
#      return Sid
# end

# """
# Diagnal tensorMap `S` -> `S^-1`.
# """
# function inv(S::TensorMap)
#      Sid = deepcopy(S)
#      for (c, blk) in blocks(Sid)
#           for ii in 1:size(blk)[1]
#                block(Sid, c)[ii, ii] = one(block(S, c)[ii, ii]) / block(S, c)[ii, ii]
#           end
#      end
#      return Sid
# end

function check_qn(ipeps::iPEPS, envs::iPEPSenv)
     Lx = envs.Lx
     Ly = envs.Ly
     for xx in 1:Lx, yy in 1:Ly
          @assert space(ipeps[xx, yy])[4] == space(ipeps[xx+1, yy])[1]' "iPEPS: [$xx, $yy] right ≠ [$(xx+1), $yy] left"
          @assert space(ipeps[xx, yy])[5] == space(ipeps[xx, yy+1])[2]' "iPEPS: [$xx, $yy] bottom ≠ [$xx, $(yy+1)] top"

          @assert space(envs[xx, yy].transfer.l)[1] == space(envs[xx, yy].corner.lt)[2]' "Env: [$xx, $yy] left transfer ≠ left-top corner"
          @assert space(envs[xx, yy].transfer.l)[2] == space(envs[xx, yy].corner.lb)[1]' "Env: [$xx, $yy] left transfer ≠ left-bottom corner"
          @assert space(envs[xx, yy].transfer.r)[3] == space(envs[xx, yy].corner.rt)[2]' "Env: [$xx, $yy] right transfer ≠ right-top corner"
          @assert space(envs[xx, yy].transfer.r)[4] == space(envs[xx, yy].corner.rb)[2]' "Env: [$xx, $yy] right transfer ≠ right-bottom corner"
          @assert space(envs[xx, yy].transfer.b)[1] == space(envs[xx, yy].corner.lb)[2]' "Env: [$xx, $yy] bottom transfer ≠ left-bottom corner"
          @assert space(envs[xx, yy].transfer.b)[4] == space(envs[xx, yy].corner.rb)[1]' "Env: [$xx, $yy] bottom transfer ≠ right-bottom corner"
          @assert space(envs[xx, yy].transfer.t)[1] == space(envs[xx, yy].corner.lt)[1]' "Env: [$xx, $yy] top transfer ≠ left-top corner"
          @assert space(envs[xx, yy].transfer.t)[2] == space(envs[xx, yy].corner.rt)[1]' "Env: [$xx, $yy] top transfer ≠ right-top corner"
     end
     return nothing
end