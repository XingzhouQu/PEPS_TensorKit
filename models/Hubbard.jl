function n(pspace::GradedSpace)
    n = isometry(pspace, pspace)
    block(n, Irrep[SU₂](0) ⊠ FermionParity(0))[1, 1] = 0
    block(n, Irrep[SU₂](0) ⊠ FermionParity(0))[2, 2] = 2
    return n
end

function nd(pspace::GradedSpace)
    nd = TensorMap(zeros, pspace, pspace)
    block(nd, Irrep[SU₂](0) ⊠ FermionParity(0))[2, 2] = 1
    return nd
end

# S⋅S interaction
function SS(pspace::GradedSpace)
    aspace = GradedSpace{fSU₂}((1 => 1))
    SL = TensorMap(ones, Float64, pspace, pspace ⊗ aspace) * sqrt(3) / 2
    SR = permute(SL', ((2, 1), (3,)))

    @tensor SS[p1, p3; p2, p4] := SL[p1, p2, a] * SR[a, p3, p4]
    return SS
end

# hopping term, FdagF
function FdagF(pspace::GradedSpace)
    aspace = GradedSpace{fSU₂}((1 // 2 => 1))
    Fdag = TensorMap(zeros, pspace, pspace ⊗ aspace)
    block(Fdag, Irrep[SU₂](1 / 2) ⊠ FermionParity(1))[1, 1] = 1.0
    block(Fdag, Irrep[SU₂](0) ⊠ FermionParity(0))[2, 1] = sqrt(2)
    F = TensorMap(zeros, aspace ⊗ pspace, pspace)
    block(F, Irrep[SU₂](1 / 2) ⊠ FermionParity(1))[1, 1] = 1.0
    block(F, Irrep[SU₂](0) ⊠ FermionParity(0))[1, 2] = -sqrt(2)

    @tensor fdagf[p1, p3; p2, p4] := Fdag[p1, p2, a] * F[a, p3, p4]
    return fdagf
end

# hopping term, FFdag
# warning: each hopping term == Fᵢ^dag Fⱼ - Fᵢ Fⱼ^dag
function FFdag(pspace::GradedSpace)
    aspace = GradedSpace{fSU₂}((1 // 2 => 1))
    iso = isometry(aspace, flip(aspace))
    Fdagtmp = TensorMap(zeros, pspace, pspace ⊗ aspace)
    block(Fdagtmp, Irrep[SU₂](1 / 2) ⊠ FermionParity(1))[1, 1] = 1.0
    block(Fdagtmp, Irrep[SU₂](0) ⊠ FermionParity(0))[2, 1] = sqrt(2)
    Ftmp = TensorMap(zeros, aspace ⊗ pspace, pspace)
    block(Ftmp, Irrep[SU₂](1 / 2) ⊠ FermionParity(1))[1, 1] = 1.0
    block(Ftmp, Irrep[SU₂](0) ⊠ FermionParity(0))[1, 2] = -sqrt(2)

    @tensor F[a; c d] := Fdagtmp'[a, b, c] * iso'[d, b]
    @tensor Fdag[d a; c] := Ftmp'[a, b, c] * iso[b, d]
    @tensor ffdag[p1, p3; p2, p4] := F[p1, p2, a] * Fdag[a, p3, p4]
    return ffdag
end


function Hubbard_hij(para::Dict{Symbol,Any})
    t = para[:t]
    U = para[:U]
    μ = para[:μ]
    pspace = GradedSpace{fSU₂}((0 => 2), (1 // 2 => 1))
    Opnd = nd(pspace)
    fdagf = FdagF(pspace)
    ffdag = FFdag(pspace)
    Opn = n(pspace)
    # 这里单点项要➗4？
    gate = -t * (fdagf + ffdag) + (U / 4) * (Opnd ⊗ id(pspace) + id(pspace) ⊗ Opnd)
    -(μ / 4) * (Opn ⊗ id(pspace) + id(pspace) ⊗ Opn)
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
        error("Unsupported tag. Check input tag or add this operator.")
    end
end