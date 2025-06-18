"""
     module Z₂SU₂tJ2orb

Prepare some commonly used objects for Z₂×SU₂ `tJ` fermions, i.e. local `d = 3^4 = 81` Hilbert space without double occupancy. 
     
consequence: xup, zup, zdn, xdn

"""

module Z₂SU₂tJ2orb

using TensorKit

# 后缀1的是子空间, 四个合成一个
const pspace1 = Rep[ℤ₂×SU₂]((0, 0) => 1, (1, 1 // 2) => 1)

const Id1 = TensorMap(ones, pspace1, pspace1)

const pspace = Rep[ℤ₂×SU₂]((0, 0) => 9, (1, 1 // 2) => 12, (0, 1) => 9, (1, 3 // 2) => 4, (0, 2) => 1)

const iso = isometry(pspace, pspace1 ⊗ pspace1 ⊗ pspace1 ⊗ pspace1)

const Z1 = let
    Z1 = TensorMap(ones, pspace1, pspace1)
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
    @tensor Zxup[p1; p2] := iso[p1, s1, s2, s3, s4] * Z1[s1, s1p] * Id1[s2, s2p] *
                            Id1[s3, s3p] * Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    Zxup
end

const Zzup = let
    @tensor Zzup[p1; p2] := iso[p1, s1, s2, s3, s4] * Id1[s1, s1p] * Z1[s2, s2p] *
                            Id1[s3, s3p] * Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    Zzup
end

const Zzdn = let
    @tensor Zzdn[p1; p2] := iso[p1, s1, s2, s3, s4] * Id1[s1, s1p] * Id1[s2, s2p] *
                            Z1[s3, s3p] * Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    Zzdn
end

const Zxdn = let
    @tensor Zxdn[p1; p2] := iso[p1, s1, s2, s3, s4] * Id1[s1, s1p] * Id1[s2, s2p] *
                            Id1[s3, s3p] * Z1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    Zxdn
end

# ====================== onsite charge operators ======================
const nxup = let
    n = TensorMap(ones, pspace1, pspace1)
    block(n, Irrep[ℤ₂×SU₂](0, 0)) .= 0
    @tensor nxup[p1; p2] := iso[p1, s1, s2, s3, s4] * n[s1, s1p] * Id1[s2, s2p] *
                            Id1[s3, s3p] * Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    nxup
end

const nzup = let
    n = TensorMap(ones, pspace1, pspace1)
    block(n, Irrep[ℤ₂×SU₂](0, 0)) .= 0
    @tensor nzup[p1; p2] := iso[p1, s1, s2, s3, s4] * Id1[s1, s1p] * n[s2, s2p] *
                            Id1[s3, s3p] * Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    nzup
end

const nzdn = let
    n = TensorMap(ones, pspace1, pspace1)
    block(n, Irrep[ℤ₂×SU₂](0, 0)) .= 0
    @tensor nzdn[p1; p2] := iso[p1, s1, s2, s3, s4] * Id1[s1, s1p] * Id1[s2, s2p] *
                            n[s3, s3p] * Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    nzdn
end

const nxdn = let
    n = TensorMap(ones, pspace1, pspace1)
    block(n, Irrep[ℤ₂×SU₂](0, 0)) .= 0
    @tensor nxdn[p1; p2] := iso[p1, s1, s2, s3, s4] * Id1[s1, s1p] * Id1[s2, s2p] *
                            Id1[s3, s3p] * n[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    nxdn
end

# ====================== on-site ZZ nn interaction ======================
const nzupnzdn = let
    n = TensorMap(ones, pspace1, pspace1)
    block(n, Irrep[ℤ₂×SU₂](0, 0)) .= 0
    @tensor nzupnzdn[p1; p2] := iso[p1, s1, s2, s3, s4] * Id1[s1, s1p] * n[s2, s2p] *
                                n[s3, s3p] * Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    nzupnzdn
end

# ====================== on-site spin interaction ======================
# S⋅S interaction, 子空间里的
const SS1 = let
    aspace = Rep[ℤ₂×SU₂]((0, 1) => 1)
    SL = TensorMap(ones, Float64, pspace1, pspace1 ⊗ aspace) * sqrt(3) / 2

    SR = permute(SL', ((2, 1), (3,)))
    SL, SR
end

const SxupSzup = let
    SL = SS1[1]
    SR = SS1[2]
    @tensor SxupSzup[p1; p2] := iso[p1, s1, s2, s3, s4] * SL[s1, s1p, a] * SR[a, s2, s2p] *
                                Id1[s3, s3p] * Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    SxupSzup
end

const SxdnSzdn = let
    SL = SS1[1]
    SR = SS1[2]
    @tensor SxdnSzdn[p1; p2] := iso[p1, s1, s2, s3, s4] * Id1[s1, s1p] * Id1[s2, s2p] *
                                SL[s3, s3p, a] * SR[a, s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    SxdnSzdn
end

const SzupSzdn = let
    SL = SS1[1]
    SR = SS1[2]
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

# zz hopping t⟂
const FdagzupFzdn = let
    Fdag = FdagF1[1]
    F = FdagF1[2]
    @tensor FdagzupFzdn[p1; p2] := iso[p1, s1, s2, s3, s4] * Id1[s1, s1p] * Id1[s4, s4p] * Z1[s2, s2in] *
                                   Fdag[s2in, s2p, a] * F[a, s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    FdagzupFzdn
end

const FzupFdagzdn = let
    F = FFdag1[1]
    Fdag = FFdag1[2]
    @tensor FzupFdagzdn[p1; p2] := iso[p1, s1, s2, s3, s4] * Id1[s1, s1p] * Id1[s4, s4p] * Z1[s2, s2in] *
                                   F[s2in, s2p, a] * Fdag[a, s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    FzupFdagzdn
end

# ====================== Nearest neighbor spin interaction =======================
const SxupSxup = let
    SL = SS1[1]
    SR = SS1[2]
    @tensor SLx[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * SL[s1, s1p, a] * Id1[s4, s4p] * Id1[s2, s2p] *
                                Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor SRx[a, p1; p2] := iso[p1, s1, s2, s3, s4] * SR[a, s1, s1p] * Id1[s4, s4p] * Id1[s2, s2p] *
                              Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    SLx, SRx
end

const SxdnSxdn = let
    SL = SS1[1]
    SR = SS1[2]
    @tensor SLx[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * SL[s4, s4p, a] * Id1[s1, s1p] * Id1[s2, s2p] *
                                Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor SRx[a, p1; p2] := iso[p1, s1, s2, s3, s4] * SR[a, s4, s4p] * Id1[s1, s1p] * Id1[s2, s2p] *
                              Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    SLx, SRx
end

# ====================== Nearest neighbor hopping term =======================
# ---------------- xx hopping ----------------
const FdagxupFxup = let
    Fdag = FdagF1[1]
    F = FdagF1[2]
    @tensor Fdagxup[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * Fdag[s1, s1p, a] * Id1[s4, s4p] * Id1[s2, s2p] *
                                    Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor Fxup[a, p1; p2] := iso[p1, s1, s2, s3, s4] * F[a, s1, s1p] * Id1[s4, s4p] * Id1[s2, s2p] *
                               Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    Fdagxup, Fxup
end

const FxupFdagxup = let
    F = FFdag1[1]
    Fdag = FFdag1[2]
    @tensor Fxup[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * F[s1, s1p, a] * Id1[s4, s4p] * Id1[s2, s2p] *
                                 Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor Fdagxup[a, p1; p2] := iso[p1, s1, s2, s3, s4] * Fdag[a, s1, s1p] * Id1[s4, s4p] * Id1[s2, s2p] *
                                  Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    Fxup, Fdagxup
end

const FdagxdnFxdn = let
    Fdag = FdagF1[1]
    F = FdagF1[2]
    @tensor Fdagxdn[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * Fdag[s4, s4p, a] * Id1[s1, s1p] * Id1[s2, s2p] *
                                    Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor Fxdn[a, p1; p2] := iso[p1, s1, s2, s3, s4] * F[a, s4, s4p] * Id1[s1, s1p] * Id1[s2, s2p] *
                               Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    Fdagxdn, Fxdn
end

const FxdnFdagxdn = let
    F = FFdag1[1]
    Fdag = FFdag1[2]
    @tensor Fxdn[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * F[s4, s4p, a] * Id1[s1, s1p] * Id1[s2, s2p] *
                                 Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor Fdagxdn[a, p1; p2] := iso[p1, s1, s2, s3, s4] * Fdag[a, s4, s4p] * Id1[s1, s1p] * Id1[s2, s2p] *
                                  Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    Fxdn, Fdagxdn
end

# ---------------- zz hopping ----------------
const FdagzupFzup = let
    Fdag = FdagF1[1]
    F = FdagF1[2]
    @tensor Fdagzup[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * Fdag[s2, s2p, a] * Id1[s4, s4p] * Id1[s1, s1p] *
                                    Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor Fzup[a, p1; p2] := iso[p1, s1, s2, s3, s4] * F[a, s2, s2p] * Id1[s4, s4p] * Id1[s1, s1p] *
                               Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    Fdagzup, Fzup
end

const FzupFdagzup = let
    F = FFdag1[1]
    Fdag = FFdag1[2]
    @tensor Fzup[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * F[s2, s2p, a] * Id1[s4, s4p] * Id1[s1, s1p] *
                                 Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor Fdagzup[a, p1; p2] := iso[p1, s1, s2, s3, s4] * Fdag[a, s2, s2p] * Id1[s4, s4p] * Id1[s1, s1p] *
                                  Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    Fzup, Fdagzup
end

const FdagzdnFzdn = let
    Fdag = FdagF1[1]
    F = FdagF1[2]
    @tensor Fdagzdn[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * Fdag[s3, s3p, a] * Id1[s1, s1p] * Id1[s2, s2p] *
                                    Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor Fzdn[a, p1; p2] := iso[p1, s1, s2, s3, s4] * F[a, s3, s3p] * Id1[s1, s1p] * Id1[s2, s2p] *
                               Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    Fdagzdn, Fzdn
end

const FzdnFdagzdn = let
    F = FFdag1[1]
    Fdag = FFdag1[2]
    @tensor Fxdn[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * F[s3, s3p, a] * Id1[s1, s1p] * Id1[s2, s2p] *
                                 Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor Fdagxdn[a, p1; p2] := iso[p1, s1, s2, s3, s4] * Fdag[a, s3, s3p] * Id1[s1, s1p] * Id1[s2, s2p] *
                                  Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    Fxdn, Fdagxdn
end

# ---------------- xz hopping ----------------
const FdagxupFzup = let
    Fdag = FdagF1[1]
    F = FdagF1[2]
    @tensor Fdagxup[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * Fdag[s1, s1p, a] * Id1[s4, s4p] * Id1[s2, s2p] *
                                    Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor Fzup[a, p1; p2] := iso[p1, s1, s2, s3, s4] * F[a, s2, s2p] * Id1[s4, s4p] * Id1[s1, s1p] *
                               Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    Fdagxup, Fzup
end

const FxupFdagzup = let
    F = FFdag1[1]
    Fdag = FFdag1[2]
    @tensor Fxup[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * F[s1, s1p, a] * Id1[s4, s4p] * Id1[s2, s2p] *
                                 Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor Fdagzup[a, p1; p2] := iso[p1, s1, s2, s3, s4] * Fdag[a, s2, s2p] * Id1[s4, s4p] * Id1[s1, s1p] *
                                  Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    Fxup, Fdagzup
end

const FdagzupFxup = let
    Fdag = FdagF1[1]
    F = FdagF1[2]
    @tensor Fdagzup[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * Fdag[s2, s2p, a] * Id1[s4, s4p] * Id1[s1, s1p] *
                                    Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor Fxup[a, p1; p2] := iso[p1, s1, s2, s3, s4] * F[a, s1, s1p] * Id1[s4, s4p] * Id1[s2, s2p] *
                               Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    Fdagzup, Fxup
end

const FzupFdagxup = let
    F = FFdag1[1]
    Fdag = FFdag1[2]
    @tensor Fzup[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * F[s2, s2p, a] * Id1[s4, s4p] * Id1[s1, s1p] *
                                 Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor Fdagxup[a, p1; p2] := iso[p1, s1, s2, s3, s4] * Fdag[a, s1, s1p] * Id1[s4, s4p] * Id1[s2, s2p] *
                                  Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    Fzup, Fdagxup
end

const FdagxdnFzdn = let
    Fdag = FdagF1[1]
    F = FdagF1[2]
    @tensor Fdagxdn[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * Fdag[s4, s4p, a] * Id1[s1, s1p] * Id1[s2, s2p] *
                                    Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor Fzdn[a, p1; p2] := iso[p1, s1, s2, s3, s4] * F[a, s3, s3p] * Id1[s4, s4p] * Id1[s1, s1p] *
                               Id1[s2, s2p] * iso'[s1p, s2p, s3p, s4p, p2]
    Fdagxdn, Fzdn
end

const FxdnFdagzdn = let
    F = FFdag1[1]
    Fdag = FFdag1[2]
    @tensor Fxdn[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * F[s4, s4p, a] * Id1[s1, s1p] * Id1[s2, s2p] *
                                 Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor Fdagzdn[a, p1; p2] := iso[p1, s1, s2, s3, s4] * Fdag[a, s3, s3p] * Id1[s4, s4p] * Id1[s2, s2p] *
                                  Id1[s1, s1p] * iso'[s1p, s2p, s3p, s4p, p2]
    Fxdn, Fdagzdn
end

const FdagzdnFxdn = let
    Fdag = FdagF1[1]
    F = FdagF1[2]
    @tensor Fdagzdn[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * Fdag[s3, s3p, a] * Id1[s1, s1p] * Id1[s2, s2p] *
                                    Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor Fxdn[a, p1; p2] := iso[p1, s1, s2, s3, s4] * F[a, s4, s4p] * Id1[s3, s3p] * Id1[s1, s1p] *
                               Id1[s2, s2p] * iso'[s1p, s2p, s3p, s4p, p2]
    Fdagzdn, Fxdn
end

const FzdnFdagxdn = let
    F = FFdag1[1]
    Fdag = FFdag1[2]
    @tensor Fzdn[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * F[s3, s3p, a] * Id1[s1, s1p] * Id1[s2, s2p] *
                                 Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor Fdagxdn[a, p1; p2] := iso[p1, s1, s2, s3, s4] * Fdag[a, s4, s4p] * Id1[s3, s3p] * Id1[s2, s2p] *
                                  Id1[s1, s1p] * iso'[s1p, s2p, s3p, s4p, p2]
    Fzdn, Fdagxdn
end

# ---------------------- interlayer onsite pairing ---------------------
# singlet pairing Δᵢⱼ^dag Δₖₗ  (onsite terms)
const Δₛdagx = let
    A = FdagF1[1]
    B = FFdag1[2]
    @tensor deltaSdag[p1; p2] := iso[p1, s1, s2, s3, s4] * Z1[s1, s1in] * A[s1in, s1p, a] * B[a, s4, s4p] *
                                 Z1[s2, s2p] * Z1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    deltaSdag
end

const Δₛx = let
    A = FFdag1[1]
    B = FdagF1[2]
    @tensor deltaS[p1; p2] := iso[p1, s1, s2, s3, s4] * Z1[s1, s1in] * A[s1in, s1p, a] * B[a, s4, s4p] *
                              Z1[s2, s2p] * Z1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    deltaS
end

const Δₛdagz = let
    A = FdagF1[1]
    B = FFdag1[2]
    @tensor deltaSdag[p1; p2] := iso[p1, s1, s2, s3, s4] * Z1[s2, s2in] * A[s2in, s2p, a] * B[a, s3, s3p] *
                                 Id1[s1, s1p] * Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    deltaSdag
end

const Δₛz = let
    A = FFdag1[1]
    B = FdagF1[2]
    @tensor deltaS[p1; p2] := iso[p1, s1, s2, s3, s4] * Z1[s2, s2in] * A[s2in, s2p, a] * B[a, s3, s3p] *
                              Id1[s1, s1p] * Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    deltaS
end

const Δₛdagxz = let
    A = FdagF1[1]
    B = FFdag1[2]
    @tensor deltaSdag[p1; p2] := iso[p1, s1, s2, s3, s4] * Z1[s1, s1in] * A[s1in, s1p, a] * B[a, s3, s3p] *
                                 Z1[s2, s2p] * Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    deltaSdag
end

const Δₛxz = let
    A = FFdag1[1]
    B = FdagF1[2]
    @tensor deltaS[p1; p2] := iso[p1, s1, s2, s3, s4] * Z1[s1, s1in] * A[s1in, s1p, a] * B[a, s3, s3p] *
                              Z1[s2, s2p] * Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    deltaS
end

const Δₛdagzx = let
    A = FdagF1[1]
    B = FFdag1[2]
    @tensor deltaSdag[p1; p2] := iso[p1, s1, s2, s3, s4] * Z1[s2, s2in] * A[s2in, s2p, a] * B[a, s4, s4p] *
                                 Z1[s3, s3p] * Id1[s1, s1p] * iso'[s1p, s2p, s3p, s4p, p2]
    deltaSdag
end

const Δₛzx = let
    A = FFdag1[1]
    B = FdagF1[2]
    @tensor deltaS[p1; p2] := iso[p1, s1, s2, s3, s4] * Z1[s2, s2in] * A[s2in, s2p, a] * B[a, s4, s4p] *
                              Z1[s3, s3p] * Id1[s1, s1p] * iso'[s1p, s2p, s3p, s4p, p2]
    deltaS
end

# ---------------------- interlayer nearest neighbor pairing ---------------------
const Δₛdagx2site = let
    A = FdagF1[1]
    B = FFdag1[2]
    @tensor Fdagxup[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * A[s1, s1p, a] * Id1[s4, s4p] * Id1[s2, s2p] *
                                    Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor Fdagxdn[a, p1; p2] := iso[p1, s1, s2, s3, s4] * B[a, s4, s4p] * Id1[s1, s1p] * Id1[s2, s2p] *
                                  Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor Δₛdagx2site[p1, p3; p2, p4] := Zxup[p1, p1in] * Fdagxup[p1in, p2, a] * Fdagxdn[a, p3, p4]
    Δₛdagx2site
end

const Δₛx2site = let
    A = FFdag1[1]
    B = FdagF1[2]
    @tensor Fxup[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * A[s1, s1p, a] * Id1[s4, s4p] * Id1[s2, s2p] *
                                 Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor Fxdn[a, p1; p2] := iso[p1, s1, s2, s3, s4] * B[a, s4, s4p] * Id1[s1, s1p] * Id1[s2, s2p] *
                               Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor Δₛx2site[p1, p3; p2, p4] := Zxup[p1, p1in] * Fxup[p1in, p2, a] * Fxdn[a, p3, p4]
    Δₛx2site
end

const Δₛdagz2site = let
    A = FdagF1[1]
    B = FFdag1[2]
    @tensor Fdagzup[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * A[s2, s2p, a] * Id1[s4, s4p] * Id1[s1, s1p] *
                                    Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor Fdagzdn[a, p1; p2] := iso[p1, s1, s2, s3, s4] * B[a, s3, s3p] * Id1[s1, s1p] * Id1[s2, s2p] *
                                  Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor Δₛdagz2site[p1, p3; p2, p4] := Zzup[p1, p1in] * Fdagzup[p1in, p2, a] * Fdagzdn[a, p3, p4]
    Δₛdagz2site
end

const Δₛz2site = let
    A = FFdag1[1]
    B = FdagF1[2]
    @tensor Fzup[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * A[s2, s2p, a] * Id1[s4, s4p] * Id1[s1, s1p] *
                                 Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor Fzdn[a, p1; p2] := iso[p1, s1, s2, s3, s4] * B[a, s3, s3p] * Id1[s1, s1p] * Id1[s2, s2p] *
                               Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor Δₛz2site[p1, p3; p2, p4] := Zzup[p1, p1in] * Fzup[p1in, p2, a] * Fzdn[a, p3, p4]
    Δₛz2site
end

const Δₛdagxz2site = let
    A = FdagF1[1]
    B = FFdag1[2]
    @tensor Fdagxup[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * A[s1, s1p, a] * Id1[s4, s4p] * Id1[s2, s2p] *
                                    Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor Fdagzdn[a, p1; p2] := iso[p1, s1, s2, s3, s4] * B[a, s3, s3p] * Id1[s1, s1p] * Id1[s2, s2p] *
                                  Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor Δₛdagxz2site[p1, p3; p2, p4] := Zxup[p1, p1in] * Fdagxup[p1in, p2, a] * Fdagzdn[a, p3, p4]
    Δₛdagxz2site
end

const Δₛxz2site = let
    A = FFdag1[1]
    B = FdagF1[2]
    @tensor Fxup[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * A[s1, s1p, a] * Id1[s4, s4p] * Id1[s2, s2p] *
                                 Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor Fzdn[a, p1; p2] := iso[p1, s1, s2, s3, s4] * B[a, s3, s3p] * Id1[s1, s1p] * Id1[s2, s2p] *
                               Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor Δₛxz2site[p1, p3; p2, p4] := Zxup[p1, p1in] * Fxup[p1in, p2, a] * Fzdn[a, p3, p4]
    Δₛxz2site
end

const Δₛdagzx2site = let
    A = FdagF1[1]
    B = FFdag1[2]
    @tensor Fdagzup[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * A[s2, s2p, a] * Id1[s4, s4p] * Id1[s1, s1p] *
                                    Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor Fdagxdn[a, p1; p2] := iso[p1, s1, s2, s3, s4] * B[a, s4, s4p] * Id1[s1, s1p] * Id1[s2, s2p] *
                                  Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor Δₛdagzx2site[p1, p3; p2, p4] := Zzup[p1, p1in] * Fdagzup[p1in, p2, a] * Fdagxdn[a, p3, p4]
    Δₛdagzx2site
end

const Δₛzx2site = let
    A = FFdag1[1]
    B = FdagF1[2]
    @tensor Fzup[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * A[s2, s2p, a] * Id1[s4, s4p] * Id1[s1, s1p] *
                                 Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor Fxdn[a, p1; p2] := iso[p1, s1, s2, s3, s4] * B[a, s4, s4p] * Id1[s1, s1p] * Id1[s2, s2p] *
                               Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor Δₛzx2site[p1, p3; p2, p4] := Zzup[p1, p1in] * Fzup[p1in, p2, a] * Fxdn[a, p3, p4]
    Δₛzx2site
end

end

"""
     const Z2SU2tJ2orb = Z₂SU₂tJ2orb
"""
const Z2SU2tJ2orb = Z₂SU₂tJ2orb


function tJ2orb_hij(para::Dict{Symbol,Any})
    tc = para[:tc]
    td = para[:td]
    tperp = para[:tperp]  # t⟂ for dz2
    Jc = para[:Jc]
    Jperp = para[:Jperp]  # J⟂ for dz2
    JH = para[:JH]  # Hund's rule coupling
    V = para[:V]  # hybridization
    εx = para[:εx]
    εz = para[:εz]

    # intralayer xx hopping
    Fdagxup, Fxup = Z2SU2tJ2orb.FdagxupFxup
    Zxup = Z2SU2tJ2orb.Zxup
    @tensor fdagfxup[p1, p3; p2, p4] := Zxup[p1, p1in] * Fdagxup[p1in, p2, a] * Fxup[a, p3, p4]
    Fxup, Fdagxup = Z2SU2tJ2orb.FxupFdagxup
    @tensor ffdagxup[p1, p3; p2, p4] := Zxup[p1, p1in] * Fxup[p1in, p2, a] * Fdagxup[a, p3, p4]

    Fdagxdn, Fxdn = Z2SU2tJ2orb.FdagxdnFxdn
    Zxdn = Z2SU2tJ2orb.Zxdn
    @tensor fdagfxdn[p1, p3; p2, p4] := Zxdn[p1, p1in] * Fdagxdn[p1in, p2, a] * Fxdn[a, p3, p4]
    Fxdn, Fdagxdn = Z2SU2tJ2orb.FxdnFdagxdn
    @tensor ffdagxdn[p1, p3; p2, p4] := Zxdn[p1, p1in] * Fxdn[p1in, p2, a] * Fdagxdn[a, p3, p4]

    # intralayer zz hopping
    Fdagzup, Fzup = Z2SU2tJ2orb.FdagzupFzup
    Zzup = Z2SU2tJ2orb.Zzup
    @tensor fdagfzup[p1, p3; p2, p4] := Zzup[p1, p1in] * Fdagzup[p1in, p2, a] * Fzup[a, p3, p4]
    Fzup, Fdagzup = Z2SU2tJ2orb.FzupFdagzup
    @tensor ffdagzup[p1, p3; p2, p4] := Zzup[p1, p1in] * Fzup[p1in, p2, a] * Fdagzup[a, p3, p4]

    Fdagzdn, Fzdn = Z2SU2tJ2orb.FdagzdnFzdn
    Zzdn = Z2SU2tJ2orb.Zzdn
    @tensor fdagfzdn[p1, p3; p2, p4] := Zzdn[p1, p1in] * Fdagzdn[p1in, p2, a] * Fzdn[a, p3, p4]
    Fzdn, Fdagzdn = Z2SU2tJ2orb.FzdnFdagzdn
    @tensor ffdagzdn[p1, p3; p2, p4] := Zzdn[p1, p1in] * Fzdn[p1in, p2, a] * Fdagzdn[a, p3, p4]

    # intralayer xz hopping
    Fdagxup, Fzup = Z2SU2tJ2orb.FdagxupFzup
    Zxup = Z2SU2tJ2orb.Zxup
    @tensor fdagfxzup[p1, p3; p2, p4] := Zxup[p1, p1in] * Fdagxup[p1in, p2, a] * Fzup[a, p3, p4]
    Fxup, Fdagzup = Z2SU2tJ2orb.FxupFdagzup
    @tensor ffdagxzup[p1, p3; p2, p4] := Zxup[p1, p1in] * Fxup[p1in, p2, a] * Fdagzup[a, p3, p4]

    Fdagzup, Fxup = Z2SU2tJ2orb.FdagzupFxup
    Zzup = Z2SU2tJ2orb.Zzup
    @tensor fdagfzxup[p1, p3; p2, p4] := Zzup[p1, p1in] * Fdagzup[p1in, p2, a] * Fxup[a, p3, p4]
    Fzup, Fdagxup = Z2SU2tJ2orb.FzupFdagxup
    @tensor ffdagzxup[p1, p3; p2, p4] := Zzup[p1, p1in] * Fzup[p1in, p2, a] * Fdagxup[a, p3, p4]

    Fdagxdn, Fzdn = Z2SU2tJ2orb.FdagxdnFzdn
    Zxdn = Z2SU2tJ2orb.Zxdn
    @tensor fdagfxzdn[p1, p3; p2, p4] := Zxdn[p1, p1in] * Fdagxdn[p1in, p2, a] * Fzdn[a, p3, p4]
    Fxdn, Fdagzdn = Z2SU2tJ2orb.FxdnFdagzdn
    @tensor ffdagxzdn[p1, p3; p2, p4] := Zxdn[p1, p1in] * Fxdn[p1in, p2, a] * Fdagzdn[a, p3, p4]

    Fdagzdn, Fxdn = Z2SU2tJ2orb.FdagzdnFxdn
    Zzdn = Z2SU2tJ2orb.Zzdn
    @tensor fdagfzxdn[p1, p3; p2, p4] := Zzdn[p1, p1in] * Fdagzdn[p1in, p2, a] * Fxdn[a, p3, p4]
    Fzdn, Fdagxdn = Z2SU2tJ2orb.FzdnFdagxdn
    @tensor ffdagzxdn[p1, p3; p2, p4] := Zzdn[p1, p1in] * Fzdn[p1in, p2, a] * Fdagxdn[a, p3, p4]

    # intralayer Sx ⋅ Sx interaction
    SLxup, SRxup = Z2SU2tJ2orb.SxupSxup
    @tensor SxupSxup[p1, p3; p2, p4] := SLxup[p1, p2, a] * SRxup[a, p3, p4]
    SLxdn, SRxdn = Z2SU2tJ2orb.SxdnSxdn
    @tensor SxdnSxdn[p1, p3; p2, p4] := SLxdn[p1, p2, a] * SRxdn[a, p3, p4]

    SxupSzup = Z2SU2tJ2orb.SxupSzup
    SxdnSzdn = Z2SU2tJ2orb.SxdnSzdn
    FdagzupFzdn = Z2SU2tJ2orb.FdagzupFzdn
    FzupFdagzdn = Z2SU2tJ2orb.FzupFdagzdn
    SzupSzdn = Z2SU2tJ2orb.SzupSzdn
    nzupnzdn = Z2SU2tJ2orb.nzupnzdn
    nxup = Z2SU2tJ2orb.nxup
    nxdn = Z2SU2tJ2orb.nxdn
    nzup = Z2SU2tJ2orb.nzup
    nzdn = Z2SU2tJ2orb.nzdn
    Id = Z2SU2tJ2orb.Id
    # 这里单点项要➗4, 注意区分杂化x/y方向符号相反
    gateNNx = -tc * (fdagfxup - ffdagxup + fdagfxdn - ffdagxdn) +
              Jc * (SxupSxup - 0.25 * Z2SU2tJ2orb.nxup ⊗ Z2SU2tJ2orb.nxup + SxdnSxdn - 0.25 * Z2SU2tJ2orb.nxdn ⊗ Z2SU2tJ2orb.nxdn) +
              -td * (fdagfzup - ffdagzup + fdagfzdn - ffdagzdn) +
              -V * 0.5 * (fdagfxzup - ffdagxzup + fdagfxzdn - ffdagxzdn + fdagfzxup - ffdagzxup + fdagfzxdn - ffdagzxdn) +
              -0.25 * JH * ((SxupSzup + SxdnSzdn) ⊗ Id + Id ⊗ (SxupSzup + SxdnSzdn)) +
              -0.25 * tperp * ((FdagzupFzdn - FzupFdagzdn) ⊗ Id + Id ⊗ (FdagzupFzdn - FzupFdagzdn)) +
              0.25 * Jperp * ((SzupSzdn - 0.25 * nzupnzdn) ⊗ Id + Id ⊗ (SzupSzdn - 0.25 * nzupnzdn)) +
              0.25 * εx * ((nxup + nxdn) ⊗ Id + Id ⊗ (nxup + nxdn)) +
              0.25 * εz * ((nzup + nzdn) ⊗ Id + Id ⊗ (nzup + nzdn))

    gateNNy = -tc * (fdagfxup - ffdagxup + fdagfxdn - ffdagxdn) +
              Jc * (SxupSxup - 0.25 * Z2SU2tJ2orb.nxup ⊗ Z2SU2tJ2orb.nxup + SxdnSxdn - 0.25 * Z2SU2tJ2orb.nxdn ⊗ Z2SU2tJ2orb.nxdn) +
              -td * (fdagfzup - ffdagzup + fdagfzdn - ffdagzdn) +
              V * 0.5 * (fdagfxzup - ffdagxzup + fdagfxzdn - ffdagxzdn + fdagfzxup - ffdagzxup + fdagfzxdn - ffdagzxdn) +
              -0.25 * JH * ((SxupSzup + SxdnSzdn) ⊗ Id + Id ⊗ (SxupSzup + SxdnSzdn)) +
              -0.25 * tperp * ((FdagzupFzdn - FzupFdagzdn) ⊗ Id + Id ⊗ (FdagzupFzdn - FzupFdagzdn)) +
              0.25 * Jperp * ((SzupSzdn - 0.25 * nzupnzdn) ⊗ Id + Id ⊗ (SzupSzdn - 0.25 * nzupnzdn)) +
              0.25 * εx * ((nxup + nxdn) ⊗ Id + Id ⊗ (nxup + nxdn)) +
              0.25 * εz * ((nzup + nzdn) ⊗ Id + Id ⊗ (nzup + nzdn))
    return gateNNx, gateNNy
end

function get_op_tJ(tag::String, para::Dict{Symbol,Any})
    ## two site Obs
    if tag == "hijNNx"
        return tJ2orb_hij(para)[1]
    elseif tag == "hijNNy"
        return tJ2orb_hij(para)[2]
    elseif tag == "Δₛx2site"
        return Z2SU2tJ2orb.Δₛx2site
    elseif tag == "Δₛdagx2site"
        return Z2SU2tJ2orb.Δₛdagx2site
    elseif tag == "Δₛz2site"
        return Z2SU2tJ2orb.Δₛz2site
    elseif tag == "Δₛdagz2site"
        return Z2SU2tJ2orb.Δₛdagz2site
    elseif tag == "Δₛxz2site"
        return Z2SU2tJ2orb.Δₛxz2site
    elseif tag == "Δₛdagxz2site"
        return Z2SU2tJ2orb.Δₛdagxz2site
    elseif tag == "Δₛzx2site"
        return Z2SU2tJ2orb.Δₛzx2site
    elseif tag == "Δₛdagzx2site"
        return Z2SU2tJ2orb.Δₛdagzx2site
        ## single site Obs
    elseif tag == "Δₛx"
        return Z2SU2tJ2orb.Δₛx
    elseif tag == "Δₛdagx"
        return Z2SU2tJ2orb.Δₛdagx
    elseif tag == "Δₛxz"
        return Z2SU2tJ2orb.Δₛxz
    elseif tag == "Δₛdagxz"
        return Z2SU2tJ2orb.Δₛdagxz
    elseif tag == "Δₛzx"
        return Z2SU2tJ2orb.Δₛzx
    elseif tag == "Δₛdagzx"
        return Z2SU2tJ2orb.Δₛdagzx
    elseif tag == "Δₛz"
        return Z2SU2tJ2orb.Δₛz
    elseif tag == "Δₛdagz"
        return Z2SU2tJ2orb.Δₛdagz
    elseif tag == "Nx"
        return Z2SU2tJ2orb.nxup + Z2SU2tJ2orb.nxdn
    elseif tag == "Nz"
        return Z2SU2tJ2orb.nzup + Z2SU2tJ2orb.nzdn
    elseif tag == "SxupSzup"
        return Z2SU2tJ2orb.SxupSzup
    elseif tag == "SxdnSzdn"
        return Z2SU2tJ2orb.SxdnSzdn
    elseif tag == "SzupSzdn"
        return Z2SU2tJ2orb.SzupSzdn
    elseif tag == "nzupnzdn"
        return Z2SU2tJ2orb.nzupnzdn
    else
        error("Unsupported tag $tag. Check input tag or add this operator.")
    end
end