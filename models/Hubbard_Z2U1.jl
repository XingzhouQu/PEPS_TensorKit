# pspace = Rep[ℤ₂×U₁]((0, 0) => 2, (1, 1/2) => 1, (1, -1/2)=> 1)

function Z(pspace::GradedSpace)
    z = TensorMap(zeros, pspace, pspace)
    block(z, Irrep[ℤ₂×U₁](1, 1 // 2)) .= -1.0
    block(z, Irrep[ℤ₂×U₁](1, -1 // 2)) .= -1.0
    block(z, Irrep[ℤ₂×U₁](0, 0))[1, 1] = 1.0
    block(z, Irrep[ℤ₂×U₁](0, 0))[2, 2] = 1.0
    return z
end

function n₊(pspace::GradedSpace)
    n₊ = TensorMap(zeros, pspace, pspace)
    block(n₊, Irrep[ℤ₂×U₁](0, 0))[2, 2] = 1.0
    block(n₊, Irrep[ℤ₂×U₁](1, 1 / 2)) .= 1.0
    return n₊
end

function n₋(pspace::GradedSpace)
    n₋ = TensorMap(zeros, pspace, pspace)
    block(n₋, Irrep[ℤ₂×U₁](0, 0))[2, 2] = 1.0
    block(n₋, Irrep[ℤ₂×U₁](1, -1 / 2)) .= 1.0
    return n₋
end

function n(pspace::GradedSpace)
    return n₊(pspace) + n₋(pspace)
end

function nd(pspace::GradedSpace)
    return n₊(pspace) * n₋(pspace)
end

function Sz(pspace::GradedSpace)
    return (n₊(pspace) - n₋(pspace)) / 2
end

# S+ S- interaction
# convention: S⋅S = SzSz + (S₊₋ + S₋₊)/2
function S₊₋(pspace::GradedSpace)
    aspace = Rep[ℤ₂×U₁]((0, 1) => 1)
    S₊ = TensorMap(ones, pspace, pspace ⊗ aspace)
    S₋ = TensorMap(ones, aspace ⊗ pspace, pspace)
    return S₊, S₋
end

function S₋₊(pspace::GradedSpace)
    Sp, Sm = S₊₋(pspace)
    aspace = Rep[ℤ₂×U₁]((0, 1) => 1)
    iso = isometry(aspace, flip(aspace))
    @tensor S₋[a; c d] := Sp'[a, b, c] * iso'[d, b]
    @tensor S₊[d a; c] := Sm'[a, b, c] * iso[b, d]
    return S₋, S₊
end

# hopping term, FdagF₊ = c↑^dag c↑
function FdagF₊(pspace::GradedSpace)
    aspace = Rep[ℤ₂×U₁]((1, 1 // 2) => 1)
    Fdag₊ = TensorMap(zeros, pspace, pspace ⊗ aspace)
    block(Fdag₊, Irrep[ℤ₂×U₁](0, 0))[1, 1] = 1.0
    block(Fdag₊, Irrep[ℤ₂×U₁](1, 1 // 2))[1, 1] = 1.0
    F₊ = TensorMap(zeros, aspace ⊗ pspace, pspace)
    block(F₊, Irrep[ℤ₂×U₁](0, 0))[1, 2] = 1.0
    block(F₊, Irrep[ℤ₂×U₁](1, 1 // 2))[2] = 1.0
    return Fdag₊, F₊
end
function FdagF₋(pspace::GradedSpace)
    # note c↓^dag|↑⟩ = -|↑↓⟩, c↓|↑↓⟩ = -|↑⟩  
    aspace = Rep[ℤ₂×U₁]((1, -1 // 2) => 1)
    Fdag₋ = TensorMap(zeros, pspace, pspace ⊗ aspace)
    block(Fdag₋, Irrep[ℤ₂×U₁](0, 0))[1, 1] = -1.0
    block(Fdag₋, Irrep[ℤ₂×U₁](1, -1 // 2))[1, 1] = 1.0
    F₋ = TensorMap(zeros, aspace ⊗ pspace, pspace)
    block(F₋, Irrep[ℤ₂×U₁](0, 0))[1, 2] = -1.0
    block(F₋, Irrep[ℤ₂×U₁](1, -1 // 2))[2] = 1.0
    return Fdag₋, F₋
end
function FFdag₊(pspace::GradedSpace)
    Fdagup, Fup = FdagF₊(pspace)
    aspace = Rep[ℤ₂×U₁]((1, 1 // 2) => 1)
    iso = isometry(aspace, flip(aspace))
    @tensor F₊[a; c d] := Fdagup'[a, b, c] * iso'[d, b]
    @tensor Fdag₊[d a; c] := Fup'[a, b, c] * iso[b, d]
    return F₊, Fdag₊
end
function FFdag₋(pspace::GradedSpace)
    Fdagdn, Fdn = FdagF₋(pspace)
    aspace = Rep[ℤ₂×U₁]((1, -1 // 2) => 1)
    iso = isometry(aspace, flip(aspace))
    @tensor F₋[a; c d] := Fdagdn'[a, b, c] * iso'[d, b]
    @tensor Fdag₋[d a; c] := Fdn'[a, b, c] * iso[b, d]
    return F₋, Fdag₋
end

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
    OpZ = Z(pspace)
    Opnd = nd(pspace)
    Opn = n(pspace)
    OpI = id(pspace)
    Fdag₊, F₊ = FdagF₊(pspace)
    @tensor fdagf₊[p1, p3; p2, p4] := OpZ[p1, p1in] * Fdag₊[p1in, p2, a] * F₊[a, p3, p4] # OpZ[p1, p1in] *
    Fdag₋, F₋ = FdagF₋(pspace)
    @tensor fdagf₋[p1, p3; p2, p4] := OpZ[p1, p1in] * Fdag₋[p1in, p2, a] * F₋[a, p3, p4]

    F₊, Fdag₊ = FFdag₊(pspace)
    @tensor ffdag₊[p1, p3; p2, p4] := OpZ[p1, p1in] * F₊[p1in, p2, a] * Fdag₊[a, p3, p4] # OpZ[p1, p1in] *
    F₋, Fdag₋ = FFdag₋(pspace)
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
        Fdag, F = FdagF₊(para[:pspace])
        OpZ = Z(para[:pspace])
        @tensor fdagf[p1, p3; p2, p4] := OpZ[p1, p1in] * Fdag[p1in, p2, a] * F[a, p3, p4]
        return fdagf
    elseif tag == "CdagCdn"
        Fdag, F = FdagF₋(para[:pspace])
        OpZ = Z(para[:pspace])
        @tensor fdagf[p1, p3; p2, p4] := OpZ[p1, p1in] * Fdag[p1in, p2, a] * F[a, p3, p4]
        return fdagf
    elseif tag == "SpSm"
        SL, SR = S₊₋(para[:pspace])
        @tensor ss[p1, p3; p2, p4] := SL[p1, p2, a] * SR[a, p3, p4]
        return ss
    elseif tag == "SzSz"
        return Sz(para[:pspace]) ⊗ Sz(para[:pspace])
    elseif tag == "NN"
        return n(para[:pspace]) ⊗ n(para[:pspace])
    elseif tag == "Nup"
        return n₊(para[:pspace])
    elseif tag == "Ndn"
        return n₋(para[:pspace])
    elseif tag == "Sz"
        return Sz(para[:pspace])
    elseif tag == "N"
        return n(para[:pspace])
    elseif tag == "Nd"
        return nd(para[:pspace])
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