module Z2SU2Fermion

using TensorKit

const pspace = Rep[ℤ₂×SU₂]((0, 0) => 2, (1, 1 // 2) => 1)

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

# S⋅S interaction
const SS = let
    aspace = Rep[ℤ₂×SU₂]((0, 1) => 1)
    SL = TensorMap(ones, Float64, pspace, pspace ⊗ aspace) * sqrt(3) / 2
    SR = permute(SL', ((2, 1), (3,)))
    SL, SR
end

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
    Fdagtmp, Ftmp = Z2SU2Fermion.FdagF
    iso = isometry(aspace, flip(aspace))
    @tensor F[a; c d] := Fdagtmp'[a, b, c] * iso'[d, b]
    @tensor Fdag[d a; c] := Ftmp'[a, b, c] * iso[b, d]
    F, Fdag
end

# singlet paring operator
const Δₛ = let
    A = Z2SU2Fermion.FdagF[1]
    aspace = Rep[ℤ₂×SU₂]((1, 1 // 2) => 1)
    iso = isometry(aspace, flip(aspace)) / sqrt(2)
    @tensor B[d a; b] := A[a b c] * iso[c d]
    C = permute(B', ((1,), (3, 2)))
    D = permute(A', ((2, 1), (3,)))
    C, D
end

const Δₛdag = let
    A = Z2SU2Fermion.FdagF[1]
    aspace = Rep[ℤ₂×SU₂]((1, 1 // 2) => 1)
    iso = isometry(aspace, flip(aspace)) / sqrt(2)
    @tensor B[d a; b] := A[a b c] * iso[c d]
    A, B
end

end

const Z₂SU₂Fermion = Z2SU2Fermion

function Hubbard_hij(para::Dict{Symbol,Any})
    t = para[:t]
    U = para[:U]
    μ = para[:μ]
    pspace = para[:pspace]
    OpZ = Z2SU2Fermion.Z
    Opnd = Z2SU2Fermion.nd
    Opn = Z2SU2Fermion.n
    OpI = isometry(pspace, pspace)
    Fdag, F = Z2SU2Fermion.FdagF
    @tensor fdagf[p1, p3; p2, p4] := OpZ[p1, p1in] * Fdag[p1in, p2, a] * F[a, p3, p4]

    F, Fdag = Z2SU2Fermion.FFdag
    @tensor ffdag[p1, p3; p2, p4] := OpZ[p1, p1in] * F[p1in, p2, a] * Fdag[a, p3, p4]
    # 这里单点项要➗4
    gate = -t * (fdagf - ffdag) + (U / 4) * (Opnd ⊗ OpI + OpI ⊗ Opnd) -
           (μ / 4) * (Opn ⊗ OpI + OpI ⊗ Opn)
    return [gate]
end

function get_op_Hubbard(tag::String, para::Dict{Symbol,Any})
    if tag == "hij"
        return Hubbard_hij(para)[1]
    elseif tag == "CdagC"
        Fdag, F = Z2SU2Fermion.FdagF
        @tensor fdagf[p1, p3; p2, p4] := Fdag[p1, p2, a] * F[a, p3, p4]
        return fdagf
    elseif tag == "SS"
        SL, SR = Z2SU2Fermion.SS
        @tensor ss[p1, p3; p2, p4] := SL[p1, p2, a] * SR[a, p3, p4]
        return ss
    elseif tag == "NN"
        return Z2SU2Fermion.n ⊗ Z2SU2Fermion.n
    elseif tag == "N"
        return Z2SU2Fermion.n
    elseif tag == "Nd"
        return Z2SU2Fermion.nd
    elseif tag == "Δₛ"
        C, D = Z2SU2Fermion.Δₛ
        @tensor deltas[p1, p3; p2, p4] := C[p1, p2, a] * D[a, p3, p4]
        return deltas
    elseif tag == "Δₛdag"
        A, B = Z2SU2Fermion.Δₛdag
        @tensor deltasdag[p1, p3; p2, p4] := A[p1, p2, a] * B[a, p3, p4]
        return deltasdag
    else
        error("Unsupported tag $tag. Check input tag or add this operator.")
    end
end