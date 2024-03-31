"""
Diagnal tensorMap `S` -> `√S^-1`
"""
function inv_sqrt!(S::TensorMap)
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


"""
`check_qn(ipeps::iPEPS, envs::iPEPSenv) \n`

Check the quantum numbers for `ipeps::iPEPS` and corresponding `envs::iPEPSenv`. \n
Warnings will be thrown if mismatched quantum numbers are detected.\n
Useful for debug.
"""
function check_qn(ipeps::iPEPS, envs::iPEPSenv)
     Lx = envs.Lx
     Ly = envs.Ly
     for xx in 1:Lx, yy in 1:Ly
          if space(ipeps[xx, yy])[4] != space(ipeps[xx+1, yy])[1]'
               @warn "iPEPS: [$xx, $yy] right $(space(ipeps[xx, yy])[4]) ≠ [$(xx+1), $yy] left $(space(ipeps[xx+1, yy])[1]')"
          end
          if space(ipeps[xx, yy])[5] != space(ipeps[xx, yy+1])[2]'
               @warn "iPEPS: [$xx, $yy] bottom $(space(ipeps[xx, yy])[5]) ≠ [$xx, $(yy+1)] top $(space(ipeps[xx, yy+1])[2]')"
          end
          if space(envs[xx, yy].transfer.l)[1] != space(envs[xx, yy].corner.lt)[2]'
               @warn "Env: [$xx, $yy] left transfer $(space(envs[xx, yy].transfer.l)[1]) ≠ left-top corner $(space(envs[xx, yy].corner.lt)[2]')"
          end
          if space(envs[xx, yy].transfer.l)[2] != space(envs[xx, yy].corner.lb)[1]'
               @warn "Env: [$xx, $yy] left transfer $(space(envs[xx, yy].transfer.l)[2]) ≠ left-bottom corner $(space(envs[xx, yy].corner.lb)[1]')"
          end
          if space(envs[xx, yy].transfer.r)[3] != space(envs[xx, yy].corner.rt)[2]'
               @warn "Env: [$xx, $yy] right transfer $(space(envs[xx, yy].transfer.r)[3]) ≠ right-top corner $(space(envs[xx, yy].corner.rt)[2]')"
          end
          if space(envs[xx, yy].transfer.r)[4] != space(envs[xx, yy].corner.rb)[2]'
               @warn "Env: [$xx, $yy] right transfer $(space(envs[xx, yy].transfer.r)[4]) ≠ right-bottom corner $(space(envs[xx, yy].corner.rb)[2]')"
          end
          if space(envs[xx, yy].transfer.b)[1] != space(envs[xx, yy].corner.lb)[2]'
               @warn "Env: [$xx, $yy] bottom transfer $(space(envs[xx, yy].transfer.b)[1]) ≠ left-bottom corner $(space(envs[xx, yy].corner.lb)[2]')"
          end
          if space(envs[xx, yy].transfer.b)[4] != space(envs[xx, yy].corner.rb)[1]'
               @warn "Env: [$xx, $yy] bottom transfer $(space(envs[xx, yy].transfer.b)[4]) ≠ right-bottom corner $(space(envs[xx, yy].corner.rb)[1]')"
          end
          if space(envs[xx, yy].transfer.t)[1] != space(envs[xx, yy].corner.lt)[1]'
               @warn "Env: [$xx, $yy] top transfer $(space(envs[xx, yy].transfer.t)[1]) ≠ left-top corner $(space(envs[xx, yy].corner.lt)[1]')"
          end
          if space(envs[xx, yy].transfer.t)[2] != space(envs[xx, yy].corner.rt)[1]'
               @warn "Env: [$xx, $yy] top transfer $(space(envs[xx, yy].transfer.t)[2]) ≠ right-top corner $(space(envs[xx, yy].corner.rt)[1]')"
          end
     end
     return nothing
end