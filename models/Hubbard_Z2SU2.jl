function Z(pspace::GradedSpace)
    Z = isometry(pspace, pspace)
    block(Z, Irrep[ℤ₂×SU₂](1, 1 // 2)) .= -1
    return Z
end

function n(pspace::GradedSpace)
    n = isometry(pspace, pspace)
    block(n, Irrep[ℤ₂×SU₂](0, 0))[1, 1] = 0
    block(n, Irrep[ℤ₂×SU₂](0, 0))[2, 2] = 2
    return n
end

function nd(pspace::GradedSpace)
    nd = TensorMap(zeros, pspace, pspace)
    block(nd, Irrep[ℤ₂×SU₂](0, 0))[2, 2] = 1
    return nd
end

# S⋅S interaction
function SS(pspace::GradedSpace)
    aspace = Rep[ℤ₂×SU₂]((0, 1) => 1)
    SL = TensorMap(ones, Float64, pspace, pspace ⊗ aspace) * sqrt(3) / 2
    SR = permute(SL', ((2, 1), (3,)))

    return SL, SR
    # @tensor SS[p1, p3; p2, p4] := SL[p1, p2, a] * SR[a, p3, p4]
    # return SS
end

# hopping term, FdagF
function FdagF(pspace::GradedSpace)
    aspace = Rep[ℤ₂×SU₂]((1, 1 // 2) => 1)
    Fdag = TensorMap(zeros, pspace, pspace ⊗ aspace)
    block(Fdag, Irrep[ℤ₂×SU₂](1, 1 // 2))[1, 1] = 1
    block(Fdag, Irrep[ℤ₂×SU₂](0, 0))[2, 1] = sqrt(2)
    F = TensorMap(zeros, aspace ⊗ pspace, pspace)
    block(F, Irrep[ℤ₂×SU₂](1, 1 // 2))[1, 1] = 1
    block(F, Irrep[ℤ₂×SU₂](0, 0))[1, 2] = -sqrt(2)

    return Fdag, F
    # @tensor fdagf[p1, p3; p2, p4] := Fdag[p1, p2, a] * F[a, p3, p4]
    # return fdagf
end

# hopping term, FFdag
# warning: each hopping term == Fᵢ^dag Fⱼ - Fᵢ Fⱼ^dag
function FFdag(pspace::GradedSpace)
    aspace = Rep[ℤ₂×SU₂]((1, 1 // 2) => 1)
    Fdagtmp, Ftmp = FdagF(pspace)
    iso = isometry(aspace, flip(aspace))
    @tensor F[a; c d] := Fdagtmp'[a, b, c] * iso'[d, b]
    @tensor Fdag[d a; c] := Ftmp'[a, b, c] * iso[b, d]

    return F, Fdag
    # @tensor ffdag[p1, p3; p2, p4] := F[p1, p2, a] * Fdag[a, p3, p4]
    # return ffdag
end

# singlet paring operator
function Δₛ(pspace::GradedSpace)
    A = FdagF(pspace)[1]
    aspace = Rep[ℤ₂×SU₂]((1, 1 // 2) => 1)
    iso = isometry(aspace, flip(aspace)) / sqrt(2)
    @tensor B[d a; b] := A[a b c] * iso[c d]
    C = permute(B', ((1,), (3, 2)))
    D = permute(A', ((2, 1), (3,)))

    return C, D
end

function Δₛdag(pspace::GradedSpace)
    A = FdagF(pspace)[1]
    aspace = Rep[ℤ₂×SU₂]((1, 1 // 2) => 1)
    iso = isometry(aspace, flip(aspace)) / sqrt(2)
    @tensor B[d a; b] := A[a b c] * iso[c d]

    return A, B
end


function Hubbard_hij(para::Dict{Symbol,Any})
    t = para[:t]
    U = para[:U]
    μ = para[:μ]
    pspace = para[:pspace]
    OpZ = Z(pspace)
    Opnd = nd(pspace)
    Opn = n(pspace)
    OpI = id(pspace)
    Fdag, F = FdagF(pspace)
    @tensor fdagf[p1, p3; p2, p4] := OpZ[p1, p1in] * Fdag[p1in, p2, a] * F[a, p3, p4]
    # @tensor fdagf[p1, p3; p2, p4] := Fdag[p1, p2, a] * F[a, p3, p4]

    F, Fdag = FFdag(pspace)
    @tensor ffdag[p1, p3; p2, p4] := OpZ[p1, p1in] * F[p1in, p2, a] * Fdag[a, p3, p4]
    # @tensor ffdag[p1, p3; p2, p4] := F[p1, p2, a] * Fdag[a, p3, p4]
    # 这里单点项要➗4
    gate = -t * (fdagf - ffdag) + (U / 4) * (Opnd ⊗ OpI + OpI ⊗ Opnd) -
           (μ / 4) * (Opn ⊗ OpI + OpI ⊗ Opn)
    return [gate]
end

function get_op_Hubbard(tag::String, para::Dict{Symbol,Any})
    if tag == "hij"
        return Hubbard_hij(para)[1]
    elseif tag == "CdagC"
        Fdag, F = FdagF(para[:pspace])
        @tensor fdagf[p1, p3; p2, p4] := Fdag[p1, p2, a] * F[a, p3, p4]
        return fdagf
    elseif tag == "SS"
        SL, SR = SS(para[:pspace])
        @tensor ss[p1, p3; p2, p4] := SL[p1, p2, a] * SR[a, p3, p4]
        return ss
    elseif tag == "NN"
        return n(para[:pspace]) ⊗ n(para[:pspace])
    elseif tag == "N"
        return n(para[:pspace])
    elseif tag == "Nd"
        return nd(para[:pspace])
    elseif tag == "Δₛ"
        C, D = Δₛ(para[:pspace])
        @tensor deltas[p1, p3; p2, p4] := C[p1, p2, a] * D[a, p3, p4]
        return deltas
    elseif tag == "Δₛdag"
        A, B = Δₛdag(para[:pspace])
        @tensor deltasdag[p1, p3; p2, p4] := A[p1, p2, a] * B[a, p3, p4]
        return deltasdag
    else
        error("Unsupported tag $tag. Check input tag or add this operator.")
    end
end