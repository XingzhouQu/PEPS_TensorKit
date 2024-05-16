import Base.getindex

"""
Convention: s1 and s2 are spaces to be swap.

For pspace, always put charge symmetry in front of spin symmetry. e.g. U1charge × SU2spin

Return space: (s1, s2) ← (s1, s2)
"""
function swap_gate(s1::T, s2::T; Eltype=Float64) where {T<:ElementarySpace}
    @assert sectortype(s1) == sectortype(s2) "Input spaces should have same sectortype."
    # -----------得到sector type------------------
    rep = collect(sectors(s1))[1]
    reptype = typeof(rep[1])  # !! 约定 pspace 总是把电荷对称性放在自旋对称性前面, 如 U1charge × SU2spin
    # -------------------------------------------
    tmp1 = id(s1)
    tmp2 = id(s2)
    gate = tmp1 ⊗ tmp2
    for (f1, f2) in fusiontrees(gate)
        # @assert f1 == f2 "Fusion and splitting sectors should match in generating swap gates."
        # 为了类型稳定，不需要改动的也乘一个1 ？
        isoddParity(f1, reptype) ? (gate[f1, f2] *= -one(Eltype)) : (gate[f1, f2] *= one(Eltype))
    end
    return gate
end

"""
Check if the input fusiontree needs a minus sign. i.e. If both uncoupled sectors have odd parity. 
"""
function isoddParity(f::FusionTree, reptype::T) where {T<:Union{Type{U1Irrep},Type{Z2Irrep},Type{SU2Irrep}}}
    # f.uncoupled isa Tuple, e.g. f.uncoupled = (Irrep[U₁ × SU₂](-1, 0), Irrep[U₁ × SU₂](2, 0))
    @assert length(f.uncoupled) == 2
    s1 = f.uncoupled[1]
    s2 = f.uncoupled[2]
    if reptype == U1Irrep
        if isodd(s1[1].charge) && isodd(s2[1].charge)
            return true
        else
            return false
        end
    elseif reptype == Z2Irrep
        if isodd(s1[1].n) && isodd(s2[1].n)
            return true
        else
            return false
        end
    elseif reptype == SU2Irrep  # 这里不知道对不对？
        if isodd(s1[1].j * 2) && isodd(s2[1].j * 2)
            return true
        else
            return false
        end
    else
        error("Not supported charge symmetry")
    end
    return nothing
end

# 仅有电荷对称性时候, rep没有索引method, 为了和多种对称性需要索引的情况保持一致，加一个trivial 的索引方法.
function getindex(rep::T, ind::Int) where {T<:Union{U1Irrep,Z2Irrep,SU2Irrep}}
    return rep
end
