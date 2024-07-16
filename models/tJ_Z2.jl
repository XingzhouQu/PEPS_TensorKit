# pspace = Rep[ℤ₂](0 => 1, 1 => 2) s.t. 0 => |0⟩, 1 => |up⟩, |dn⟩
module Z2tJFermion

using TensorKit

const pspace = Rep[ℤ₂](0 => 1, 1 => 2)

const Z = let
    z = TensorMap(zeros, pspace, pspace)
    block(z, Irrep[ℤ₂](0)) .= 1.0
    block(z, Irrep[ℤ₂](1))[1, 1] = -1.0
    block(z, Irrep[ℤ₂](1))[2, 2] = -1.0
    z
end

const n₊ = let
    n₊ = TensorMap(zeros, pspace, pspace)
    block(n₊, Irrep[ℤ₂](1))[1, 1] = 1.0
    n₊
end

const n₋ = let
    n₋ = TensorMap(zeros, pspace, pspace)
    block(n₋, Irrep[ℤ₂](1))[2, 2] = 1.0
    n₋
end

const n = n₊ + n₋

const Sz = (n₊ - n₋) / 2

# S+ S- interaction
# convention: S⋅S = SzSz + (S₊₋ + S₋₊)/2
const S₊₋ = let
    aspace = Rep[ℤ₂](0 => 1)
    S₊ = TensorMap(zeros, pspace, pspace ⊗ aspace)
    block(S₊, Irrep[ℤ₂](1))[1, 2] = 1.0
    S₋ = TensorMap(zeros, aspace ⊗ pspace, pspace)
    block(S₋, Irrep[ℤ₂](1))[2, 1] = 1.0
    S₊, S₋
end

const S₋₊ = let
    Sp, Sm = Z2tJFermion.S₊₋
    aspace = Rep[ℤ₂](0 => 1)
    iso = isometry(aspace, flip(aspace))
    @tensor S₋[a; c d] := Sp'[a, b, c] * iso'[d, b]
    @tensor S₊[d a; c] := Sm'[a, b, c] * iso[b, d]
    S₋, S₊
end

# hopping term, FdagF₊ = c↑^dag c↑
const FdagF₊ = let
    aspace = Rep[ℤ₂](1 => 1)
    Fdag₊ = TensorMap(zeros, pspace, pspace ⊗ aspace)
    block(Fdag₊, Irrep[ℤ₂](1))[1] = 1.0
    F₊ = TensorMap(zeros, aspace ⊗ pspace, pspace)
    block(F₊, Irrep[ℤ₂](1))[1] = 1.0
    Fdag₊, F₊
end
const FdagF₋ = let
    # note c↓^dag|↑⟩ = -|↑↓⟩, c↓|↑↓⟩ = -|↑⟩  
    aspace = Rep[ℤ₂](1 => 1)
    Fdag₋ = TensorMap(zeros, pspace, pspace ⊗ aspace)
    block(Fdag₋, Irrep[ℤ₂](1))[2] = 1.0
    F₋ = TensorMap(zeros, aspace ⊗ pspace, pspace)
    block(F₋, Irrep[ℤ₂](1))[2] = 1.0
    Fdag₋, F₋
end
const FFdag₊ = let
    Fdagup, Fup = FdagF₊
    aspace = Rep[ℤ₂](1 => 1)
    iso = isometry(aspace, flip(aspace))
    @tensor F₊[a; c d] := Fdagup'[a, b, c] * iso'[d, b]
    @tensor Fdag₊[d a; c] := Fup'[a, b, c] * iso[b, d]
    F₊, Fdag₊
end
const FFdag₋ = let
    Fdagdn, Fdn = FdagF₋
    aspace = Rep[ℤ₂](1 => 1)
    iso = isometry(aspace, flip(aspace))
    @tensor F₋[a; c d] := Fdagdn'[a, b, c] * iso'[d, b]
    @tensor Fdag₋[d a; c] := Fdn'[a, b, c] * iso[b, d]
    F₋, Fdag₋
end

# singlet paring operator Δₛ = (c↑c↓ - c↓c↑)/√2
const Δₛ = let
    A = FFdag₊[1]
    B = FdagF₋[2]
    C = FFdag₋[1]
    D = FdagF₊[2]
    @tensor updn[p1, p3; p2, p4] := Z[p1, p1in] * A[p1in, p2, a] * B[a, p3, p4]
    @tensor dnup[p1, p3; p2, p4] := Z[p1, p1in] * C[p1in, p2, a] * D[a, p3, p4]

    (updn - dnup) / sqrt(2)
end
# singlet paring operator Δₛdag = (c↓dag c↑dag - c↑dag c↓dag)/√2
const Δₛdag = let
    A = FdagF₋[1]
    B = FFdag₊[2]
    C = FdagF₊[1]
    D = FFdag₋[2]
    @tensor dnup[p1, p3; p2, p4] := Z[p1, p1in] * A[p1in, p2, a] * B[a, p3, p4]
    @tensor updn[p1, p3; p2, p4] := Z[p1, p1in] * C[p1in, p2, a] * D[a, p3, p4]

    (dnup - updn) / sqrt(2)
end

end

const Z₂tJFermion = Z2tJFermion


function tJ_hij(para::Dict{Symbol,Any})
    t = para[:t]
    tp = para[:tp]  # t' for NNN hopping
    J = para[:J]
    Jp = para[:Jp]  # J' for NNN hopping
    h = para[:h]  # Zeeman field
    μ = para[:μ]
    pspace = para[:pspace]

    OpZ = Z2tJFermion.Z
    OpSz = Z2tJFermion.Sz
    Opn = Z2tJFermion.n
    OpI = isometry(pspace, pspace)

    Fdag₊, F₊ = Z2tJFermion.FdagF₊
    @tensor fdagf₊[p1, p3; p2, p4] := OpZ[p1, p1in] * Fdag₊[p1in, p2, a] * F₊[a, p3, p4]
    Fdag₋, F₋ = Z2tJFermion.FdagF₋
    @tensor fdagf₋[p1, p3; p2, p4] := OpZ[p1, p1in] * Fdag₋[p1in, p2, a] * F₋[a, p3, p4]

    F₊, Fdag₊ = Z2tJFermion.FFdag₊
    @tensor ffdag₊[p1, p3; p2, p4] := OpZ[p1, p1in] * F₊[p1in, p2, a] * Fdag₊[a, p3, p4]
    F₋, Fdag₋ = Z2tJFermion.FFdag₋
    @tensor ffdag₋[p1, p3; p2, p4] := OpZ[p1, p1in] * F₋[p1in, p2, a] * Fdag₋[a, p3, p4]

    SL, SR = Z2tJFermion.S₊₋
    @tensor spsm[p1, p3; p2, p4] := SL[p1, p2, a] * SR[a, p3, p4]
    SL, SR = Z2tJFermion.S₋₊
    @tensor smsp[p1, p3; p2, p4] := SL[p1, p2, a] * SR[a, p3, p4]

    # 这里单点项要➗4
    gateNN = -t * (fdagf₊ - ffdag₊ + fdagf₋ - ffdag₋) + J * (OpSz ⊗ OpSz + (spsm + smsp) / 2 - 0.25 * Opn ⊗ Opn) -
             (μ / 4) * (Opn ⊗ OpI + OpI ⊗ Opn) - (h / 4) * (OpSz ⊗ OpI + OpI ⊗ OpSz)
    gateNNN = -tp * (fdagf₊ - ffdag₊ + fdagf₋ - ffdag₋) + Jp * (OpSz ⊗ OpSz + (spsm + smsp) / 2 - 0.25 * Opn ⊗ Opn)
    return [gateNN, gateNNN]
end

function get_op_tJ(tag::String, para::Dict{Symbol,Any})
    if tag == "hijNN"
        return tJ_hij(para)[1]
    elseif tag == "hijNNN"
        return tJ_hij(para)[2]
    elseif tag == "CdagCup"
        Fdag, F = Z2tJFermion.FdagF₊
        OpZ = Z2tJFermion.Z
        @tensor fdagf[p1, p3; p2, p4] := OpZ[p1, p1in] * Fdag[p1in, p2, a] * F[a, p3, p4]
        return fdagf
    elseif tag == "CdagCdn"
        Fdag, F = Z2tJFermion.FdagF₋
        OpZ = Z2tJFermion.Z
        @tensor fdagf[p1, p3; p2, p4] := OpZ[p1, p1in] * Fdag[p1in, p2, a] * F[a, p3, p4]
        return fdagf
    elseif tag == "SpSm"
        SL, SR = Z2tJFermion.S₊₋
        @tensor ss[p1, p3; p2, p4] := SL[p1, p2, a] * SR[a, p3, p4]
        return ss
    elseif tag == "SzSz"
        return Z2tJFermion.Sz ⊗ Z2tJFermion.Sz
    elseif tag == "NN"
        return Z2tJFermion.n ⊗ Z2tJFermion.n
    elseif tag == "Nup"
        return Z2tJFermion.n₊
    elseif tag == "Ndn"
        return Z2tJFermion.n₋
    elseif tag == "Sz"
        return Z2tJFermion.Sz
    elseif tag == "N"
        return Z2tJFermion.n
    elseif tag == "Δₛ"
        return Z2tJFermion.Δₛ
    elseif tag == "Δₛdag"
        return Z2tJFermion.Δₛdag
    else
        error("Unsupported tag $tag. Check input tag or add this operator.")
    end
end