function n(pspace::GradedSpace)
    n = TensorMap(ones, pspace, pspace)
    block(n, Irrep[U₁×SU₂](2, 0)) .= 2
    block(n, Irrep[U₁×SU₂](0, 0)) .= 0
    return n
end

function nd(pspace::GradedSpace)
    nd = TensorMap(zeros, pspace, pspace)
    block(nd, Irrep[U₁×SU₂](2, 0)) .= 1
    return nd
end

# S⋅S interaction
function SS(pspace::GradedSpace)
    aspace = Rep[U₁×SU₂]((0, 1) => 1)
    SL = TensorMap(ones, Float64, pspace, pspace ⊗ aspace) * sqrt(3) / 2
    SR = permute(SL', ((2, 1), (3,)))

    @tensor SS[p1, p3; p2, p4] := SL[p1, p2, a] * SR[a, p3, p4]
    return SS
end

# hopping term, FdagF
function FdagF(pspace::GradedSpace)
    aspace = Rep[U₁×SU₂]((1, 1 // 2) => 1)
    Fdag = TensorMap(ones, pspace, pspace ⊗ aspace)
    block(Fdag, Irrep[U₁×SU₂](2, 0)) .= -sqrt(2)
    F = TensorMap(ones, aspace ⊗ pspace, pspace)
    block(F, Irrep[U₁×SU₂](2, 0)) .= sqrt(2)

    @tensor fdagf[p1, p3; p2, p4] := Fdag[p1, p2, a] * F[a, p3, p4]
    return fdagf
end

# hopping term, FFdag
# warning: each hopping term == Fᵢ^dag Fⱼ - Fᵢ Fⱼ^dag
function FFdag(pspace::GradedSpace)
    aspace = Rep[U₁×SU₂]((1, 1 // 2) => 1)
    Fdagtmp = TensorMap(ones, pspace, pspace ⊗ aspace)
    block(Fdagtmp, Irrep[U₁×SU₂](2, 0)) .= -sqrt(2)
    Ftmp = TensorMap(ones, aspace ⊗ pspace, pspace)
    block(Ftmp, Irrep[U₁×SU₂](2, 0)) .= sqrt(2)

    @tensor F[a; c d] := Fdagtmp'[a, b, c] * iso'[d, b]
    @tensor Fdag[d a; c] := Ftmp'[a, b, c] * iso[b, d]
    @tensor ffdag[p1, p3; p2, p4] := F[p1, p2, a] * Fdag[a, p3, p4]
    return ffdag
end


function Hubbard_hij(para::Dict{Symbol,Any})
    t = para[:t]
    U = para[:U]
    pspace = para[:pspace]
    Opnd = nd(pspace)
    fdagf = FdagF(pspace)
    ffdag = FFdag(pspace)
    OpI = isometry(pspace, pspace)
    # 这里单点项要➗4？
    gate = -t * (fdagf + ffdag) + (U / 4) * (Opnd ⊗ OpI + OpI ⊗ Opnd)
    # -(μ / 4) * (Opn ⊗ OpI + OpI ⊗ Opn)
    return [gate]
end

function get_op_Hubbard(tag::String, para::Dict{Symbol,Any})
    if tag == "hij"
        return Hubbard_hij(para)[1]
    elseif tag == "CdagC"
        return FdagF(para[:pspace])
    elseif tag == "SS"
        return SS(para[:pspace])
    elseif tag == "NN"
        return n(para[:pspace]) ⊗ n(para[:pspace])
    elseif tag == "N"
        return n(para[:pspace])
    elseif tag == "Nd"
        return nd(para[:pspace])
    else
        error("Unsupported tag $tag. Check input tag or add this operator.")
    end
end