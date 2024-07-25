"""
     module U₁SU₂tJFermion

Prepare some commonly used objects for U₁×SU₂ `tJ` fermions, i.e. local `d = 3` Hilbert space without double occupancy. 
     
Behaviors of all operators are the same as `U₁SU₂Fermion` up to the projection, details please see `U₁SU₂Fermion`. 
"""
module Z₂SU₂tJFermion

using TensorKit

const pspace = Rep[ℤ₂×SU₂]((0, 0) => 1, (1, 1 // 2) => 1)
const Z = let
    Z = TensorMap(ones, pspace, pspace)
    block(Z, Irrep[ℤ₂×SU₂](1, 1 / 2)) .= -1.0
    Z
end

const n = let
    n = TensorMap(ones, pspace, pspace)
    block(n, Irrep[ℤ₂×SU₂](0, 0)) .= 0
    n
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
    aspace = Rep[ℤ₂×SU₂]((1, 1 / 2) => 1)
    Fdag = TensorMap(ones, pspace, pspace ⊗ aspace)
    block(Fdag, Irrep[ℤ₂×SU₂](0, 0)) .= 0.0
    F = TensorMap(ones, aspace ⊗ pspace, pspace)
    block(F, Irrep[ℤ₂×SU₂](0, 0)) .= 0.0
    Fdag, F
end

# hopping term, FFdag
# warning: each hopping term == Fᵢ^dag Fⱼ - Fᵢ Fⱼ^dag
const FFdag = let

    aspace = Rep[ℤ₂×SU₂]((1, 1 / 2) => 1)
    iso = isometry(aspace, flip(aspace))
    @tensor F[a; c d] := FdagF[1]'[a, b, c] * iso'[d, b]
    @tensor Fdag[d a; c] := FdagF[2]'[a, b, c] * iso[b, d]

    F, Fdag
end

# singlet pairing Δᵢⱼ^dag Δₖₗ
const Δₛdag = let
    A = FdagF[1]
    B = FFdag[2]
    @tensor deltaSdag[p1, p3; p2, p4] := Z[p1, p1in] * A[p1in, p2, a] * B[a, p3, p4]
    deltaSdag
end

const Δₛ = let
    A = FFdag[1]
    B = FdagF[2]
    @tensor deltaS[p1, p3; p2, p4] := Z[p1, p1in] * A[p1in, p2, a] * B[a, p3, p4]
    deltaS
end

end

"""
     const Z2SU2tJFermion = Z₂SU₂tJFermion
"""
const Z2SU2tJFermion = Z₂SU₂tJFermion



function tJ_hij(para::Dict{Symbol,Any})
    t = para[:t]
    tp = para[:tp]  # t' for NNN hopping
    J = para[:J]
    Jp = para[:Jp]  # J' for NNN hopping
    μ = para[:μ]
    pspace = para[:pspace]

    OpZ = Z2SU2tJFermion.Z
    Opn = Z2SU2tJFermion.n
    OpI = isometry(pspace, pspace)

    Fdag, F = Z2SU2tJFermion.FdagF
    @tensor fdagf[p1, p3; p2, p4] := OpZ[p1, p1in] * Fdag[p1in, p2, a] * F[a, p3, p4]

    F, Fdag = Z2SU2tJFermion.FFdag
    @tensor ffdag[p1, p3; p2, p4] := OpZ[p1, p1in] * F[p1in, p2, a] * Fdag[a, p3, p4]

    SL, SR = Z2SU2tJFermion.SS
    @tensor ss[p1, p3; p2, p4] := SL[p1, p2, a] * SR[a, p3, p4]

    # 这里单点项要➗4
    gateNN = -t * (fdagf - ffdag) + J * (ss - 0.25 * Opn ⊗ Opn) - (μ / 4) * (Opn ⊗ OpI + OpI ⊗ Opn)
    gateNNN = -tp * (fdagf - ffdag) #+ Jp * (OpSz ⊗ OpSz + (spsm + smsp) / 2 - 0.25 * Opn ⊗ Opn)
    return gateNN, gateNNN
end

function get_op_tJ(tag::String, para::Dict{Symbol,Any})
    if tag == "hijNN"
        return tJ_hij(para)[1]
    elseif tag == "hijNNN"
        return tJ_hij(para)[2]
    elseif tag == "CdagC"
        Fdag, F = Z2SU2tJFermion.FdagF
        OpZ = Z2SU2tJFermion.Z
        @tensor fdagf[p1, p3; p2, p4] := OpZ[p1, p1in] * Fdag[p1in, p2, a] * F[a, p3, p4]
        return fdagf
    elseif tag == "SS"
        SL, SR = Z2SU2tJFermion.SS
        @tensor ss[p1, p3; p2, p4] := SL[p1, p2, a] * SR[a, p3, p4]
        return ss
    elseif tag == "NN"
        return Z2SU2tJFermion.n ⊗ Z2SU2tJFermion.n
    elseif tag == "N"
        return Z2SU2tJFermion.n
    elseif tag == "Δₛ"
        return Z2SU2tJFermion.Δₛ
    elseif tag == "Δₛdag"
        return Z2SU2tJFermion.Δₛdag
    else
        error("Unsupported tag $tag. Check input tag or add this operator.")
    end
end