"""
     module Z₂SU₂tJ2orb

Prepare some commonly used objects for Z₂×SU₂ `tJ` fermions, i.e. local `d = 3*2*2*3 = 36` Hilbert space without double occupancy. 
     
consequence: xup, zup, zdn, xdn

"""

module Z₂SU₂tJspin2orb

using TensorKit

# 后缀1的是子空间, 四个合成一个
const pspace1 = Rep[ℤ₂×SU₂]((0, 0) => 1, (1, 1 // 2) => 1)

const pspace2 = Rep[ℤ₂×SU₂]((1, 1 // 2) => 1)

const Id1 = TensorMap(ones, pspace1, pspace1)

const Id2 = TensorMap(ones, pspace2, pspace2)

const pspace = Rep[ℤ₂×SU₂]((0, 0) => 3, (0, 1) => 4, (1, 1 / 2) => 4, (0, 2) => 1, (1, 3 / 2) => 2)

const iso = isometry(pspace, pspace1 ⊗ pspace2 ⊗ pspace2 ⊗ pspace1)

const Z1 = let
    Z1 = TensorMap(ones, pspace1, pspace1)
    block(Z1, Irrep[ℤ₂×SU₂](1, 1 / 2)) .= -1.0
    Z1
end

const Z2 = let
    Z1 = TensorMap(ones, pspace2, pspace2)
    block(Z1, Irrep[ℤ₂×SU₂](1, 1 / 2)) .= -1.0
    Z1
end

const Id = let
    Id = TensorMap(zeros, pspace, pspace)
    for ii in 1:size(block(Id, Irrep[ℤ₂×SU₂](0, 0)), 1)
        block(Id, Irrep[ℤ₂×SU₂](0, 0))[ii, ii] = 1.0
    end
    for ii in 1:size(block(Id, Irrep[ℤ₂×SU₂](1, 1 / 2)), 1)
        block(Id, Irrep[ℤ₂×SU₂](1, 1 / 2))[ii, ii] = 1.0
    end
    for ii in 1:size(block(Id, Irrep[ℤ₂×SU₂](0, 1)), 1)
        block(Id, Irrep[ℤ₂×SU₂](0, 1))[ii, ii] = 1.0
    end
    for ii in 1:size(block(Id, Irrep[ℤ₂×SU₂](1, 3 / 2)), 1)
        block(Id, Irrep[ℤ₂×SU₂](1, 3 / 2))[ii, ii] = 1.0
    end
    for ii in 1:size(block(Id, Irrep[ℤ₂×SU₂](0, 2)), 1)
        block(Id, Irrep[ℤ₂×SU₂](0, 2))[ii, ii] = 1.0
    end
    Id
end

const Zxup = let
    @tensor Zxup[p1; p2] := iso[p1, s1, s2, s3, s4] * Z1[s1, s1p] * Id2[s2, s2p] *
                            Id2[s3, s3p] * Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    Zxup
end

const Zzup = let
    @tensor Zzup[p1; p2] := iso[p1, s1, s2, s3, s4] * Id1[s1, s1p] * Z2[s2, s2p] *
                            Id2[s3, s3p] * Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    Zzup
end

const Zzdn = let
    @tensor Zzdn[p1; p2] := iso[p1, s1, s2, s3, s4] * Id1[s1, s1p] * Id2[s2, s2p] *
                            Z2[s3, s3p] * Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    Zzdn
end

const Zxdn = let
    @tensor Zxdn[p1; p2] := iso[p1, s1, s2, s3, s4] * Id1[s1, s1p] * Id2[s2, s2p] *
                            Id2[s3, s3p] * Z1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    Zxdn
end

# ====================== onsite charge operators ======================
const nxup = let
    n = TensorMap(ones, pspace1, pspace1)
    block(n, Irrep[ℤ₂×SU₂](0, 0)) .= 0
    @tensor nxup[p1; p2] := iso[p1, s1, s2, s3, s4] * n[s1, s1p] * Id2[s2, s2p] *
                            Id2[s3, s3p] * Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    nxup
end

const nxdn = let
    n = TensorMap(ones, pspace1, pspace1)
    block(n, Irrep[ℤ₂×SU₂](0, 0)) .= 0
    @tensor nxdn[p1; p2] := iso[p1, s1, s2, s3, s4] * Id1[s1, s1p] * Id2[s2, s2p] *
                            Id2[s3, s3p] * n[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    nxdn
end

# ====================== on-site spin interaction ======================
# S⋅S interaction, 子空间里的
const SS1 = let
    aspace = Rep[ℤ₂×SU₂]((0, 1) => 1)
    SL = TensorMap(ones, Float64, pspace1, pspace1 ⊗ aspace) * sqrt(3) / 2

    SR = permute(SL', ((2, 1), (3,)))
    SL, SR
end

const SS2 = let
    aspace = Rep[ℤ₂×SU₂]((0, 1) => 1)
    SL = TensorMap(ones, Float64, pspace2, pspace2 ⊗ aspace) * sqrt(3) / 2

    SR = permute(SL', ((2, 1), (3,)))
    SL, SR
end

const SxupSzup = let
    SL = SS1[1]
    SR = SS2[2]
    @tensor SxupSzup[p1; p2] := iso[p1, s1, s2, s3, s4] * SL[s1, s1p, a] * SR[a, s2, s2p] *
                                Id2[s3, s3p] * Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    SxupSzup
end

const SxdnSzdn = let
    SL = SS2[1]
    SR = SS1[2]
    @tensor SxdnSzdn[p1; p2] := iso[p1, s1, s2, s3, s4] * Id1[s1, s1p] * Id2[s2, s2p] *
                                SL[s3, s3p, a] * SR[a, s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    SxdnSzdn
end

const SzupSzdn = let
    SL = SS2[1]
    SR = SS2[2]
    @tensor SzupSzdn[p1; p2] := iso[p1, s1, s2, s3, s4] * Id1[s1, s1p] * Id1[s4, s4p] *
                                SL[s2, s2p, a] * SR[a, s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    SzupSzdn
end

# ====================== onsite hopping term, FzdagFz ======================
const FdagF1 = let
    aspace = Rep[ℤ₂×SU₂]((1, 1 / 2) => 1)
    Fdag = TensorMap(ones, pspace1, pspace1 ⊗ aspace)
    block(Fdag, Irrep[ℤ₂×SU₂](0, 0)) .= 0.0
    F = TensorMap(ones, aspace ⊗ pspace1, pspace1)
    block(F, Irrep[ℤ₂×SU₂](0, 0)) .= 0.0
    Fdag, F
end

# hopping term, FFdag
# warning: each hopping term == Fᵢ^dag Fⱼ - Fᵢ Fⱼ^dag
const FFdag1 = let
    aspace = Rep[ℤ₂×SU₂]((1, 1 / 2) => 1)
    iso = isometry(aspace, flip(aspace))
    @tensor F[a; c d] := FdagF1[1]'[a, b, c] * iso'[d, b]
    @tensor Fdag[d a; c] := FdagF1[2]'[a, b, c] * iso[b, d]
    F, Fdag
end

# ====================== Nearest neighbor spin interaction =======================
const SxupSxup = let
    SL = SS1[1]
    SR = SS1[2]
    @tensor SLx[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * SL[s1, s1p, a] * Id1[s4, s4p] * Id2[s2, s2p] *
                                Id2[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor SRx[a, p1; p2] := iso[p1, s1, s2, s3, s4] * SR[a, s1, s1p] * Id1[s4, s4p] * Id2[s2, s2p] *
                              Id2[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    SLx, SRx
end

const SxdnSxdn = let
    SL = SS1[1]
    SR = SS1[2]
    @tensor SLx[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * SL[s4, s4p, a] * Id1[s1, s1p] * Id2[s2, s2p] *
                                Id2[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor SRx[a, p1; p2] := iso[p1, s1, s2, s3, s4] * SR[a, s4, s4p] * Id1[s1, s1p] * Id2[s2, s2p] *
                              Id2[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    SLx, SRx
end

# ====================== Nearest neighbor hopping term =======================
# ---------------- xx hopping ----------------
const FdagxupFxup = let
    Fdag = FdagF1[1]
    F = FdagF1[2]
    @tensor Fdagxup[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * Fdag[s1, s1p, a] * Id1[s4, s4p] * Id2[s2, s2p] *
                                    Id2[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor Fxup[a, p1; p2] := iso[p1, s1, s2, s3, s4] * F[a, s1, s1p] * Id1[s4, s4p] * Id2[s2, s2p] *
                               Id2[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    Fdagxup, Fxup
end

const FxupFdagxup = let
    F = FFdag1[1]
    Fdag = FFdag1[2]
    @tensor Fxup[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * F[s1, s1p, a] * Id1[s4, s4p] * Id2[s2, s2p] *
                                 Id2[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor Fdagxup[a, p1; p2] := iso[p1, s1, s2, s3, s4] * Fdag[a, s1, s1p] * Id1[s4, s4p] * Id2[s2, s2p] *
                                  Id2[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    Fxup, Fdagxup
end

const FdagxdnFxdn = let
    Fdag = FdagF1[1]
    F = FdagF1[2]
    @tensor Fdagxdn[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * Fdag[s4, s4p, a] * Id1[s1, s1p] * Id2[s2, s2p] *
                                    Id2[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor Fxdn[a, p1; p2] := iso[p1, s1, s2, s3, s4] * F[a, s4, s4p] * Id1[s1, s1p] * Id2[s2, s2p] *
                               Id2[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    Fdagxdn, Fxdn
end

const FxdnFdagxdn = let
    F = FFdag1[1]
    Fdag = FFdag1[2]
    @tensor Fxdn[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * F[s4, s4p, a] * Id1[s1, s1p] * Id2[s2, s2p] *
                                 Id2[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor Fdagxdn[a, p1; p2] := iso[p1, s1, s2, s3, s4] * Fdag[a, s4, s4p] * Id1[s1, s1p] * Id2[s2, s2p] *
                                  Id2[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    Fxdn, Fdagxdn
end

# ---------------------- interlayer pairing ---------------------
# singlet pairing Δᵢⱼ^dag Δₖₗ  (onsite terms)
const Δₛdagx = let
    A = FdagF1[1]
    B = FFdag1[2]
    @tensor deltaSdag[p1; p2] := iso[p1, s1, s2, s3, s4] * Z1[s1, s1in] * A[s1in, s1p, a] * B[a, s4, s4p] *
                                 Z2[s2, s2p] * Z2[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    deltaSdag
end

const Δₛx = let
    A = FFdag1[1]
    B = FdagF1[2]
    @tensor deltaS[p1; p2] := iso[p1, s1, s2, s3, s4] * Z1[s1, s1in] * A[s1in, s1p, a] * B[a, s4, s4p] *
                              Z2[s2, s2p] * Z2[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    deltaS
end

end

"""
     const Z2SU2tJspin2orb = Z₂SU₂tJspin2orb
"""
const Z2SU2tJspin2orb = Z₂SU₂tJspin2orb


function tJ2orb_hij(para::Dict{Symbol,Any})
    tc = para[:tc]
    Jc = para[:Jc]
    Jperp = para[:Jperp]  # J⟂ for dz2
    JH = para[:JH]  # Hund's rule coupling
    εx = para[:εx]

    # intralayer xx hopping
    Fdagxup, Fxup = Z2SU2tJspin2orb.FdagxupFxup
    Zxup = Z2SU2tJspin2orb.Zxup
    @tensor fdagfxup[p1, p3; p2, p4] := Zxup[p1, p1in] * Fdagxup[p1in, p2, a] * Fxup[a, p3, p4]
    Fxup, Fdagxup = Z2SU2tJspin2orb.FxupFdagxup
    @tensor ffdagxup[p1, p3; p2, p4] := Zxup[p1, p1in] * Fxup[p1in, p2, a] * Fdagxup[a, p3, p4]

    Fdagxdn, Fxdn = Z2SU2tJspin2orb.FdagxdnFxdn
    Zxdn = Z2SU2tJspin2orb.Zxdn
    @tensor fdagfxdn[p1, p3; p2, p4] := Zxdn[p1, p1in] * Fdagxdn[p1in, p2, a] * Fxdn[a, p3, p4]
    Fxdn, Fdagxdn = Z2SU2tJspin2orb.FxdnFdagxdn
    @tensor ffdagxdn[p1, p3; p2, p4] := Zxdn[p1, p1in] * Fxdn[p1in, p2, a] * Fdagxdn[a, p3, p4]

    # intralayer Sx ⋅ Sx interaction
    SLxup, SRxup = Z2SU2tJspin2orb.SxupSxup
    @tensor SxupSxup[p1, p3; p2, p4] := SLxup[p1, p2, a] * SRxup[a, p3, p4]
    SLxdn, SRxdn = Z2SU2tJspin2orb.SxdnSxdn
    @tensor SxdnSxdn[p1, p3; p2, p4] := SLxdn[p1, p2, a] * SRxdn[a, p3, p4]

    SxupSzup = Z2SU2tJspin2orb.SxupSzup
    SxdnSzdn = Z2SU2tJspin2orb.SxdnSzdn
    SzupSzdn = Z2SU2tJspin2orb.SzupSzdn
    nxup = Z2SU2tJspin2orb.nxup
    nxdn = Z2SU2tJspin2orb.nxdn
    Id = Z2SU2tJspin2orb.Id
    # 这里单点项要➗4, 注意区分杂化x/y方向符号相反
    gateNN = -tc * (fdagfxup - ffdagxup + fdagfxdn - ffdagxdn) +
             Jc * (SxupSxup - 0.25 * Z2SU2tJspin2orb.nxup ⊗ Z2SU2tJspin2orb.nxup + SxdnSxdn - 0.25 * Z2SU2tJspin2orb.nxdn ⊗ Z2SU2tJspin2orb.nxdn) +
             -0.25 * JH * ((SxupSzup + SxdnSzdn) ⊗ Id + Id ⊗ (SxupSzup + SxdnSzdn)) +
             0.25 * Jperp * ((SzupSzdn) ⊗ Id + Id ⊗ (SzupSzdn)) +
             0.25 * εx * ((nxup + nxdn) ⊗ Id + Id ⊗ (nxup + nxdn))

    return gateNN, gateNN
end

function get_op_tJ(tag::String, para::Dict{Symbol,Any})
    if tag == "hijNN"
        return tJ2orb_hij(para)[1]
    elseif tag == "hijNN"
        return tJ2orb_hij(para)[2]
    elseif tag == "SxSx"
        SLxup, SRxup = Z2SU2tJspin2orb.SxupSxup
        @tensor SxupSxup[p1, p3; p2, p4] := SLxup[p1, p2, a] * SRxup[a, p3, p4]
        SLxdn, SRxdn = Z2SU2tJspin2orb.SxdnSxdn
        @tensor SxdnSxdn[p1, p3; p2, p4] := SLxdn[p1, p2, a] * SRxdn[a, p3, p4]
        return SxupSxup + SxdnSxdn
        # elseif tag == "CdagC"
        #     Fdag, F = Z2SU2tJFermion.FdagF
        #     OpZ = Z2SU2tJFermion.Z
        #     @tensor fdagf[p1, p3; p2, p4] := OpZ[p1, p1in] * Fdag[p1in, p2, a] * F[a, p3, p4]
        #     return fdagf
        # elseif tag == "SS"
        #     SL, SR = Z2SU2tJFermion.SS
        #     @tensor ss[p1, p3; p2, p4] := SL[p1, p2, a] * SR[a, p3, p4]
        #     return ss
        # elseif tag == "NN"
        #     return Z2SU2tJFermion.n ⊗ Z2SU2tJFermion.n
        # elseif tag == "N"
        #     return Z2SU2tJFermion.n
    elseif tag == "Δₛx"
        return Z2SU2tJspin2orb.Δₛx
    elseif tag == "Δₛdagx"
        return Z2SU2tJspin2orb.Δₛdagx
    elseif tag == "Nx"
        return Z2SU2tJspin2orb.nxup + Z2SU2tJspin2orb.nxdn
    else
        error("Unsupported tag $tag. Check input tag or add this operator.")
    end
end