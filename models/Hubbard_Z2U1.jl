# pspace = Rep[ℤ₂×U₁]((0, 0) => 2, (1, 1/2) => 1, (1, -1/2)=> 1)
# 顺序：|0⟩, |↑↓⟩, |↑⟩, |↓⟩
# 写算符时候分清domain和co-domain中的 aspace和pspace就知道如何映射了
module Z2U1Fermion

using TensorKit

const pspace = Rep[ℤ₂×U₁]((0, 0) => 2, (1, 1/2) => 1, (1, -1/2)=> 1)

const Z = let
    z = TensorMap(zeros, pspace, pspace)
    block(z, Irrep[ℤ₂×U₁](1, 1 // 2)) .= -1.0
    block(z, Irrep[ℤ₂×U₁](1, -1 // 2)) .= -1.0
    block(z, Irrep[ℤ₂×U₁](0, 0))[1, 1] = 1.0
    block(z, Irrep[ℤ₂×U₁](0, 0))[2, 2] = 1.0
    z
end

const n₊ = let
    n₊ = TensorMap(zeros, pspace, pspace)
    block(n₊, Irrep[ℤ₂×U₁](0, 0))[2, 2] = 1.0
    block(n₊, Irrep[ℤ₂×U₁](1, 1 / 2)) .= 1.0
    n₊
end

const n₋ = let
    n₋ = TensorMap(zeros, pspace, pspace)
    block(n₋, Irrep[ℤ₂×U₁](0, 0))[2, 2] = 1.0
    block(n₋, Irrep[ℤ₂×U₁](1, -1 / 2)) .= 1.0
    n₋
end

const n = n₊ + n₋

const nd = n₊ * n₋

const Sz = (n₊ - n₋) / 2

# S+ S- interaction
# convention: S⋅S = SzSz + (S₊₋ + S₋₊)/2
const S₊₋ = let
    aspace = Rep[ℤ₂×U₁]((0, 1) => 1)
    S₊ = TensorMap(ones, pspace, pspace ⊗ aspace)
    S₋ = TensorMap(ones, aspace ⊗ pspace, pspace)
    S₊, S₋
end

const S₋₊ = let
    Sp, Sm = S₊₋
    aspace = Rep[ℤ₂×U₁]((0, 1) => 1)
    iso = isometry(aspace, flip(aspace))
    @tensor S₋[a; c d] := Sp'[a, b, c] * iso'[d, b]
    @tensor S₊[d a; c] := Sm'[a, b, c] * iso[b, d]
    S₋, S₊
end

# hopping term, FdagF₊ = c↑^dag c↑
const FdagF₊ = let
    aspace = Rep[ℤ₂×U₁]((1, 1 // 2) => 1)
    Fdag₊ = TensorMap(zeros, pspace, pspace ⊗ aspace)
    block(Fdag₊, Irrep[ℤ₂×U₁](0, 0))[2] = 1.0
    block(Fdag₊, Irrep[ℤ₂×U₁](1, 1 // 2))[1, 1] = 1.0
    F₊ = TensorMap(zeros, aspace ⊗ pspace, pspace)
    block(F₊, Irrep[ℤ₂×U₁](0, 0))[1, 2] = 1.0
    block(F₊, Irrep[ℤ₂×U₁](1, 1 // 2))[1] = 1.0
    Fdag₊, F₊
end
const FdagF₋ = let
    # note c↓^dag|↑⟩ = -|↑↓⟩, c↓|↑↓⟩ = -|↑⟩  
    aspace = Rep[ℤ₂×U₁]((1, -1 // 2) => 1)
    Fdag₋ = TensorMap(zeros, pspace, pspace ⊗ aspace)
    block(Fdag₋, Irrep[ℤ₂×U₁](0, 0))[2] = -1.0
    block(Fdag₋, Irrep[ℤ₂×U₁](1, -1 // 2))[1, 1] = 1.0
    F₋ = TensorMap(zeros, aspace ⊗ pspace, pspace)
    block(F₋, Irrep[ℤ₂×U₁](0, 0))[1, 2] = -1.0
    block(F₋, Irrep[ℤ₂×U₁](1, -1 // 2))[1] = 1.0
    Fdag₋, F₋
end
const FFdag₊ = let
    Fdagup, Fup = FdagF₊
    aspace = Rep[ℤ₂×U₁]((1, 1 // 2) => 1)
    iso = isometry(aspace, flip(aspace))
    @tensor F₊[a; c d] := Fdagup'[a, b, c] * iso'[d, b]
    @tensor Fdag₊[d a; c] := Fup'[a, b, c] * iso[b, d]
    F₊, Fdag₊
end
const FFdag₋ = let
    Fdagdn, Fdn = FdagF₋
    aspace = Rep[ℤ₂×U₁]((1, -1 // 2) => 1)
    iso = isometry(aspace, flip(aspace))
    @tensor F₋[a; c d] := Fdagdn'[a, b, c] * iso'[d, b]
    @tensor Fdag₋[d a; c] := Fdn'[a, b, c] * iso[b, d]
    F₋, Fdag₋
end

end

const Z₂U₁Fermion = Z2U1Fermion
# singlet paring operator
# function Δₛ(pspace::GradedSpace)
#     A = FdagF(pspace)[1]
#     aspace = Rep[ℤ₂×SU₂]((1, 1 // 2) => 1)
#     iso = isometry(aspace, flip(aspace)) / sqrt(2)
#     @tensor B[d a; b] := A[a b c] * iso[c d]
#     C = permute(B', ((1,), (3, 2)))
#     D = permute(A', ((2, 1), (3,)))

#     return C, D
# end

# function Δₛdag(pspace::GradedSpace)
#     A = FdagF(pspace)[1]
#     aspace = Rep[ℤ₂×SU₂]((1, 1 // 2) => 1)
#     iso = isometry(aspace, flip(aspace)) / sqrt(2)
#     @tensor B[d a; b] := A[a b c] * iso[c d]

#     return A, B
# end


function Hubbard_hij(para::Dict{Symbol,Any})
    t = para[:t]
    U = para[:U]
    μ = para[:μ]
    pspace = para[:pspace]
    OpZ = Z2U1Fermion.Z
    Opnd = Z2U1Fermion.nd
    Opn = Z2U1Fermion.n
    OpI = isometry(pspace, pspace)
    Fdag₊, F₊ = Z2U1Fermion.FdagF₊
    @tensor fdagf₊[p1, p3; p2, p4] := OpZ[p1, p1in] * Fdag₊[p1in, p2, a] * F₊[a, p3, p4] # OpZ[p1, p1in] *
    Fdag₋, F₋ = Z2U1Fermion.FdagF₋
    @tensor fdagf₋[p1, p3; p2, p4] := OpZ[p1, p1in] * Fdag₋[p1in, p2, a] * F₋[a, p3, p4]

    F₊, Fdag₊ = Z2U1Fermion.FFdag₊
    @tensor ffdag₊[p1, p3; p2, p4] := OpZ[p1, p1in] * F₊[p1in, p2, a] * Fdag₊[a, p3, p4] # OpZ[p1, p1in] *
    F₋, Fdag₋ = Z2U1Fermion.FFdag₋
    @tensor ffdag₋[p1, p3; p2, p4] := OpZ[p1, p1in] * F₋[p1in, p2, a] * Fdag₋[a, p3, p4]
    # 这里单点项要➗4
    gate = -t * (fdagf₊ - ffdag₊ + fdagf₋ - ffdag₋) + (U / 4) * (Opnd ⊗ OpI + OpI ⊗ Opnd) -
           (μ / 4) * (Opn ⊗ OpI + OpI ⊗ Opn)
    return [gate]
end

function get_op_Hubbard(tag::String, para::Dict{Symbol,Any})
    if tag == "hij"
        return Hubbard_hij(para)[1]
    elseif tag == "CdagCup"
        Fdag, F = Z2U1Fermion.FdagF₊
        OpZ = Z2U1Fermion.Z
        @tensor fdagf[p1, p3; p2, p4] := OpZ[p1, p1in] * Fdag[p1in, p2, a] * F[a, p3, p4]
        return fdagf
    elseif tag == "CdagCdn"
        Fdag, F = Z2U1Fermion.FdagF₋
        OpZ = Z2U1Fermion.Z
        @tensor fdagf[p1, p3; p2, p4] := OpZ[p1, p1in] * Fdag[p1in, p2, a] * F[a, p3, p4]
        return fdagf
    elseif tag == "SpSm"
        SL, SR = Z2U1Fermion.S₊₋
        @tensor ss[p1, p3; p2, p4] := SL[p1, p2, a] * SR[a, p3, p4]
        return ss
    elseif tag == "SzSz"
        return Z2U1Fermion.Sz ⊗ Z2U1Fermion.Sz
    elseif tag == "NN"
        return Z2U1Fermion.n ⊗ Z2U1Fermion.n
    elseif tag == "Nup"
        return Z2U1Fermion.n₊
    elseif tag == "Ndn"
        return Z2U1Fermion.n₋
    elseif tag == "Sz"
        return Z2U1Fermion.Sz
    elseif tag == "N"
        return Z2U1Fermion.n
    elseif tag == "Nd"
        return Z2U1Fermion.nd
    elseif tag == "Δₛ"
        # C, D = Δₛ(para[:pspace])
        # @tensor deltas[p1, p3; p2, p4] := C[p1, p2, a] * D[a, p3, p4]
        # return deltas
    elseif tag == "Δₛdag"
        # A, B = Δₛdag(para[:pspace])
        # @tensor deltasdag[p1, p3; p2, p4] := A[p1, p2, a] * B[a, p3, p4]
        # return deltasdag
    else
        error("Unsupported tag $tag. Check input tag or add this operator.")
    end
end