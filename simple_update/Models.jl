# hopping term, FdagF
const FdagF = let
    aspace = Rep[ℤ₂×SU₂]((1, 1 // 2) => 1)
    Fdag = TensorMap(zeros, pspace, pspace ⊗ aspace)
    block(Fdag, Irrep[ℤ₂×SU₂](1, 1 // 2))[1, 1] = 1
    block(Fdag, Irrep[ℤ₂×SU₂](0, 0))[2, 1] = sqrt(2)
    F = TensorMap(zeros, aspace ⊗ pspace, pspace)
    block(F, Irrep[ℤ₂×SU₂](1, 1 // 2))[1, 1] = 1
    block(F, Irrep[ℤ₂×SU₂](0, 0))[1, 2] = -sqrt(2)

    Fdag, F
end

# hopping term, FFdag
# warning: each hopping term == Fᵢ^dag Fⱼ - Fᵢ Fⱼ^dag
const FFdag = let
    aspace = Rep[ℤ₂×SU₂]((1, 1 // 2) => 1)
    iso = isometry(aspace, flip(aspace))
    @tensor F[a; c d] := FdagF[1]'[a, b, c] * iso'[d, b]
    @tensor Fdag[d a; c] := FdagF[2]'[a, b, c] * iso[b, d]
    F, Fdag
end

const Z = let
    Z = isometry(pspace, pspace)
    block(Z, Irrep[ℤ₂×SU₂](1, 1 // 2)) .= -1
    Z
end

const n = let
    n = isometry(pspace, pspace)
    block(n, Irrep[ℤ₂×SU₂](0, 0))[1, 1] = 0
    block(n, Irrep[ℤ₂×SU₂](0, 0))[2, 2] = 2
    n
end

const nd = let
    nd = TensorMap(zeros, pspace, pspace)
    block(nd, Irrep[ℤ₂×SU₂](0, 0))[2, 2] = 1
    nd
end

function Hubbard_gate(t::Number, U::Number)
    pspace = Rep[ℤ₂×SU₂]((1, 0) => 2, (0, 1 // 2) => 1)
    aspace = Rep[ℤ₂×SU₂]((1, 1 / 2) => 1)
    Fdag = TensorMap(ones, pspace, pspace ⊗ aspace)
    block(Fdag, Irrep[ℤ₂×SU₂](1, 0)) .= -sqrt(2)
    F = TensorMap(ones, aspace ⊗ pspace, pspace)
    block(F, Irrep[ℤ₂×SU₂](1, 0)) .= sqrt(2)

end