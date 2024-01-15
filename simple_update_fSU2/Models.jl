function FdagF(pspace::GradedSpace)
    aspace = GradedSpace{fSU₂}((1 // 2 => 1))
    Fdag = TensorMap(zeros, pspace, pspace ⊗ aspace)
    for (key, val) in blocks(Fdag)
        if key == Irrep[SU₂](0) ⊠ FermionParity(0)
            val[2, 1] = sqrt(2)
        elseif key == Irrep[SU₂](1 / 2) ⊠ FermionParity(1)
            val[1, 1] = one(val[1, 1])
        else
            nothing
        end
    end
    F = TensorMap(zeros, aspace ⊗ pspace, pspace)
    for (key, val) in blocks(F)
        if key == Irrep[SU₂](0) ⊠ FermionParity(0)
            val[1, 2] = -sqrt(2)
        elseif key == Irrep[SU₂](1 / 2) ⊠ FermionParity(1)
            val[1, 1] = one(val[1, 1])
        else
            nothing
        end
    end
    @tensor fdagf[p1, p4; p2, p3] := Fdag[p1, p2, a] * F[a, p3, p4]
    return fdagf
end


# hopping term, FFdag
# warning: each hopping term == Fᵢ^dag Fⱼ - Fᵢ Fⱼ^dag
function FFdag(pspace::GradedSpace)
    aspace = GradedSpace{fSU₂}((1 // 2 => 1))
    iso = isometry(aspace, flip(aspace))
    Fdagtmp = TensorMap(zeros, pspace, pspace ⊗ aspace)
    for (key, val) in blocks(Fdagtmp)
        if key == Irrep[SU₂](0) ⊠ FermionParity(0)
            val[2, 1] = sqrt(2)
        elseif key == Irrep[SU₂](1 / 2) ⊠ FermionParity(1)
            val[1, 1] = one(val[1, 1])
        else
            nothing
        end
    end
    Ftmp = TensorMap(zeros, aspace ⊗ pspace, pspace)
    for (key, val) in blocks(Ftmp)
        if key == Irrep[SU₂](0) ⊠ FermionParity(0)
            val[1, 2] = -sqrt(2)
        elseif key == Irrep[SU₂](1 / 2) ⊠ FermionParity(1)
            val[1, 1] = one(val[1, 1])
        else
            nothing
        end
    end
    @tensor F[a; c d] := Fdagtmp'[a, b, c] * iso'[d, b]
    @tensor Fdag[d a; c] := Ftmp'[a, b, c] * iso[b, d]
    @tensor ffdag[p1, p4; p2, p3] := F[p2, p1, a] * Fdag[a, p3, p4]
    return ffdag
end

function n(pspace::GradedSpace)
    n = isometry(pspace, pspace)
    for (key, val) in blocks(n)
        if key == Irrep[SU₂](0) ⊠ FermionParity(0)
            val[1, 1] = zero(val[1, 1])
            val[2, 2] = 2 * one(val[2, 2])
        else
            nothing
        end
    end
    return n
end

function nd(pspace::GradedSpace)
    nd = TensorMap(zeros, pspace, pspace)
    for (key, val) in blocks(nd)
        if key == Irrep[SU₂](0) ⊠ FermionParity(0)
            val[2, 2] = one(val[2, 2])
        else
            nothing
        end
    end
    return nd
end


function Hubbard_gate(t::Number, U::Number)
    pspace = GradedSpace{fSU₂}((0 => 2), (1 // 2 => 1))
    Opnd = nd(pspace)
    Opn = n(pspace)
    fdagf = FdagF(pspace)
    ffdag = FFdag(pspace)
    @show space(fdagf)
    @show space(ffdag)
    @show space(Opnd ⊗ id(pspace))
    @show space(id(pspace) ⊗ Opnd)
    gate = -t * (fdagf + ffdag) + U * (Opnd ⊗ id(pspace) + id(pspace) ⊗ Opnd)
    return gate
end