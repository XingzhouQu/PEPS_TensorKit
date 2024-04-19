using LinearAlgebra
using Statistics
import TensorKit.normalize!
using JLD2
import JLD2.save

# Save and load ipeps and envs follow the JLD2.jl convention.
# JLD2 methods are:
# using JLD2
# save("/home/tcmp2/JuliaProjects/testSave.jld2", "ipeps", ipeps, "envs", envs)
# ipeps, envs = load("/home/tcmp2/JuliaProjects/testSave.jld2", "ipeps", "envs")

function save(ipeps::iPEPS, envs::iPEPSenv, Dir::String)
     println("Saving ipeps and envs into Dir $Dir")
     save(Dir, "ipeps", ipeps, "envs", envs)
     return nothing
end

function save(ipeps::iPEPSΓΛ, Dir::String)
     println("Saving ipeps (ΓΛ form) into Dir $Dir")
     save(Dir, "ipeps", ipeps)
     return nothing
end


"""
Modified form TensorKit/src/tensors/linalg.jl

TensorMap `S` -> `√S^-1` and normalize properly.
For diag matrix only.
"""
function SqrtInv(t::AbstractTensorMap; truncErr=1e-8)
     t = t / norm(t)
     TP = eltype(t)
     cod = codomain(t)
     dom = domain(t)
     for c in union(blocksectors(cod), blocksectors(dom))
          blockdim(cod, c) == blockdim(dom, c) ||
               throw(SpaceMismatch("codomain $cod and domain $dom are not isomorphic: no inverse"))
     end
     if sectortype(t) === Trivial
          tp = diag(block(t, Trivial()))
          for (ind, val) in enumerate(tp)
               val < truncErr ? tp[ind] = zero(TP) : tp[ind] = one(TP) / sqrt(val)
          end
          rslt = TensorMap(diagm(tp), domain(t) ← codomain(t))
          return rslt / mean(tp)
     else
          data = empty(t.data)
          for (c, b) in blocks(t)
               bp = diag(b)
               for (ind, val) in enumerate(bp)
                    val < truncErr ? bp[ind] = zero(TP) : bp[ind] = one(TP) / sqrt(val)
               end
               data[c] = diagm(bp)
          end
          rslt = TensorMap(data, domain(t) ← codomain(t))
          return rslt / mean(convert(Array, rslt))
     end
end

"""
return √S. For diag SVD spectrum only.
"""
function sqrt4diag(t::AbstractTensorMap)
     cod = codomain(t)
     dom = domain(t)
     for c in union(blocksectors(cod), blocksectors(dom))
          blockdim(cod, c) == blockdim(dom, c) ||
               throw(SpaceMismatch("codomain $cod and domain $dom are not isomorphic: no inverse"))
     end
     if sectortype(t) === Trivial
          tp = diag(block(t, Trivial()))
          return TensorMap(diagm(sqrt.(tp)), domain(t) ← codomain(t))
     else
          data = empty(t.data)
          for (c, b) in blocks(t)
               bp = diag(b)
               data[c] = diagm(sqrt.(bp))
          end
          return TensorMap(data, domain(t) ← codomain(t))
     end
end


"""
print the max and min value in SVD spectrum.\n
Useful for test numerical stability.
"""
function printMinMax(S::TensorMap)
     s = convert(Array, S)
     println("SVD spectrum: Max $(maximum(s)). Min $(minimum(s)). Size $(size(s)). #of zeros $(count(i->(i==0), diag(s)))")
     return nothing
end

"""
Normaliza the ipeps.
"""
function normalize!(ipeps::iPEPS, p::Real=2)
     Lx = ipeps.Lx
     Ly = ipeps.Ly
     for xx in 1:Lx, yy in 1:Ly
          normalize!(ipeps[xx, yy], p)
     end
     return nothing
end


"""
Check the quantum numbers for `ipeps::iPEPS` and corresponding `envs::iPEPSenv`. \n
Warnings will be thrown if mismatched quantum numbers are detected.\n
Useful for debug.
"""
function check_qn(ipeps::iPEPS, envs::iPEPSenv)
     Lx = ipeps.Lx
     Ly = ipeps.Ly
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

function check_qn(ipeps::iPEPSΓΛ)
     Lx = ipeps.Lx
     Ly = ipeps.Ly
     for xx in 1:Lx, yy in 1:Ly
          if space(ipeps[xx, yy].Γ)[1] != space(ipeps[xx, yy].l)[2]'
               @warn "iPEPS: [$xx, $yy] left $(space(ipeps[xx, yy].Γ)[1]) ≠ left bond space $(space(ipeps[xx, yy].l)[2]')"
          end
          if space(ipeps[xx, yy].Γ)[2] != space(ipeps[xx, yy].t)[2]'
               @warn "iPEPS: [$xx, $yy] top $(space(ipeps[xx, yy].Γ)[2]) ≠ top bond space $(space(ipeps[xx, yy].t)[2]')"
          end
          if space(ipeps[xx, yy].Γ)[4] != space(ipeps[xx, yy].r)[1]'
               @warn "iPEPS: [$xx, $yy] right $(space(ipeps[xx, yy].Γ)[4]) ≠ right bond space $(space(ipeps[xx, yy].r)[1]')"
          end
          if space(ipeps[xx, yy].Γ)[5] != space(ipeps[xx, yy].b)[1]'
               @warn "iPEPS: [$xx, $yy] bottom $(space(ipeps[xx, yy].Γ)[5]) ≠ bottom bond space $(space(ipeps[xx, yy].b)[1]')"
          end
     end
     return nothing
end