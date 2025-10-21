"""
     module Z₂U₁tJ2orb

Prepare some commonly used objects for Z₂×U₁ `tJ` fermions, i.e. local `d = 3^4 = 81` Hilbert space without double occupancy. 
     
consequence: xup, zup, zdn, xdn. 
`x` and `z` denote the dx2-y2 orbital and dz2 orbitals. `up` and `dn` mark the upper and lower layers.

"""

module Z₂U₁tJ2orb

using TensorKit

# 后缀1的是子空间, 四个合成一个
const pspace1 = Rep[ℤ₂×U₁]((0, 0) => 1, (1, 1 / 2) => 1, (1, -1 / 2) => 1)

const Id1 = TensorMap(ones, pspace1, pspace1)

const pspace = space = Rep[ℤ₂×U₁]((0, 0) => 19, (1, 1 / 2) => 16, (0, 1) => 10, (1, -1 / 2) => 16, (0, -1) => 10, (1, 3 / 2) => 4, (0, 2) => 1, (1, -3 / 2) => 4, (0, -2) => 1)

const iso = isometry(pspace, pspace1 ⊗ pspace1 ⊗ pspace1 ⊗ pspace1)

const Z1 = let
    z = TensorMap(zeros, pspace1, pspace1)
    block(z, Irrep[ℤ₂×U₁](1, 1 // 2)) .= -1.0
    block(z, Irrep[ℤ₂×U₁](1, -1 // 2)) .= -1.0
    block(z, Irrep[ℤ₂×U₁](0, 0))[1, 1] = 1.0
    z
end

const Id = id(pspace)

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

const n₊1 = let
    n₊1 = TensorMap(zeros, pspace1, pspace1)
    block(n₊1, Irrep[ℤ₂×U₁](1, 1 / 2)) .= 1.0
    n₊1
end

const n₋1 = let
    n₋1 = TensorMap(zeros, pspace1, pspace1)
    block(n₋1, Irrep[ℤ₂×U₁](1, -1 / 2)) .= 1.0
    n₋1
end

const n1 = n₊1 + n₋1

const Sz1 = (n₊1 - n₋1) / 2

# ====================== onsite charge and spin operators ======================
# subscript ₊ and ₋ in the name of operators mean spin up and down. 
const n₊xup = let
    @tensor n₊xup[p1; p2] := iso[p1, s1, s2, s3, s4] * n₊1[s1, s1p] * Id1[s2, s2p] *
                             Id1[s3, s3p] * Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    n₊xup
end

const n₋xup = let
    @tensor n₋xup[p1; p2] := iso[p1, s1, s2, s3, s4] * n₋1[s1, s1p] * Id1[s2, s2p] *
                             Id1[s3, s3p] * Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    n₋xup
end

const nxup = n₊xup + n₋xup

const Sᶻxup = (n₊xup - n₋xup) / 2

const n₊zup = let
    @tensor n₊zup[p1; p2] := iso[p1, s1, s2, s3, s4] * Id1[s1, s1p] * n₊1[s2, s2p] *
                             Id1[s3, s3p] * Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    n₊zup
end

const n₋zup = let
    @tensor n₋zup[p1; p2] := iso[p1, s1, s2, s3, s4] * Id1[s1, s1p] * n₋1[s2, s2p] *
                             Id1[s3, s3p] * Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    n₋zup
end

const nzup = n₊zup + n₋zup

const Sᶻzup = (n₊zup - n₋zup) / 2

const n₊zdn = let
    @tensor n₊zdn[p1; p2] := iso[p1, s1, s2, s3, s4] * Id1[s1, s1p] * Id1[s2, s2p] *
                             n₊1[s3, s3p] * Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    n₊zdn
end

const n₋zdn = let
    @tensor n₋zdn[p1; p2] := iso[p1, s1, s2, s3, s4] * Id1[s1, s1p] * Id1[s2, s2p] *
                             n₋1[s3, s3p] * Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    n₋zdn
end

const nzdn = n₊zdn + n₋zdn

const Sᶻzdn = (n₊zdn - n₋zdn) / 2

const n₊xdn = let
    @tensor n₊xdn[p1; p2] := iso[p1, s1, s2, s3, s4] * Id1[s1, s1p] * Id1[s2, s2p] *
                             Id1[s3, s3p] * n₊1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    n₊xdn
end

const n₋xdn = let
    @tensor n₋xdn[p1; p2] := iso[p1, s1, s2, s3, s4] * Id1[s1, s1p] * Id1[s2, s2p] *
                             Id1[s3, s3p] * n₋1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    n₋xdn
end

const nxdn = n₊xdn + n₋xdn

const Sᶻxdn = (n₊xdn - n₋xdn) / 2

# ====================== on-site ZZ nn interaction ======================
const nzupnzdn = let
    @tensor nzupnzdn[p1; p2] := iso[p1, s1, s2, s3, s4] * Id1[s1, s1p] * n1[s2, s2p] *
                                n1[s3, s3p] * Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    nzupnzdn
end

# ====================== on-site spin interaction ======================
# S⋅S = Sᶻ⋅Sᶻ + (S₊⋅S₋ + S₋⋅S₊)/2 interaction, 子空间里的
const S₊₋1 = let
    aspace = Rep[ℤ₂×U₁]((0, 1) => 1)
    S₊ = TensorMap(ones, pspace1, pspace1 ⊗ aspace)
    S₋ = TensorMap(ones, aspace ⊗ pspace1, pspace1)
    S₊, S₋
end

const S₋₊1 = let
    Sp, Sm = S₊₋1
    aspace = Rep[ℤ₂×U₁]((0, 1) => 1)
    iso = isometry(aspace, flip(aspace))
    @tensor S₋[a; c d] := Sp'[a, b, c] * iso'[d, b]
    @tensor S₊[d a; c] := Sm'[a, b, c] * iso[b, d]
    S₋, S₊
end

const SS1 = let
    SL, SR = S₊₋1
    @tensor spsm[p1, p3; p2, p4] := SL[p1, p2, a] * SR[a, p3, p4]
    SL, SR = S₋₊1
    @tensor smsp[p1, p3; p2, p4] := SL[p1, p2, a] * SR[a, p3, p4]

    Sz1 ⊗ Sz1 + (spsm + smsp) / 2
end

const SxupSzup = let
    @tensor SxupSzup[p1; p2] := iso[p1, s1, s2, s3, s4] * SS1[s1, s2, s1p, s2p] *
                                Id1[s3, s3p] * Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    SxupSzup
end

const SxdnSzdn = let
    @tensor SxdnSzdn[p1; p2] := iso[p1, s1, s2, s3, s4] * Id1[s1, s1p] * Id1[s2, s2p] *
                                SS1[s3, s4, s3p, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    SxdnSzdn
end

const SzupSzdn = let
    @tensor SzupSzdn[p1; p2] := iso[p1, s1, s2, s3, s4] * Id1[s1, s1p] * Id1[s4, s4p] *
                                SS1[s2, s3, s2p, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    SzupSzdn
end

const SxupSxdn = let
    @tensor SxupSxdn[p1; p2] := iso[p1, s1, s2, s3, s4] * Id1[s2, s2p] * Id1[s3, s3p] *
                                SS1[s1, s4, s1p, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    SxupSxdn
end

# ====================== onsite hopping term, FzdagFz ======================
# hopping term, FdagF₊ = c↑^dag c↑
# warning: each hopping term == Fᵢ^dag Fⱼ - Fᵢ Fⱼ^dag
const FdagF₊1 = let
    aspace = Rep[ℤ₂×U₁]((1, 1 // 2) => 1)
    Fdag₊ = TensorMap(zeros, pspace1, pspace1 ⊗ aspace)
    block(Fdag₊, Irrep[ℤ₂×U₁](1, 1 // 2))[1, 1] = 1.0
    F₊ = TensorMap(zeros, aspace ⊗ pspace1, pspace1)
    block(F₊, Irrep[ℤ₂×U₁](1, 1 // 2))[1] = 1.0
    Fdag₊, F₊
end
const FdagF₋1 = let
    # note c↓^dag|↑⟩ = -|↑↓⟩, c↓|↑↓⟩ = -|↑⟩  
    aspace = Rep[ℤ₂×U₁]((1, -1 // 2) => 1)
    Fdag₋ = TensorMap(zeros, pspace1, pspace1 ⊗ aspace)
    block(Fdag₋, Irrep[ℤ₂×U₁](1, -1 // 2))[1, 1] = 1.0
    F₋ = TensorMap(zeros, aspace ⊗ pspace1, pspace1)
    block(F₋, Irrep[ℤ₂×U₁](1, -1 // 2))[1] = 1.0
    Fdag₋, F₋
end
const FFdag₊1 = let
    Fdagup, Fup = FdagF₊1
    aspace = Rep[ℤ₂×U₁]((1, 1 // 2) => 1)
    iso = isometry(aspace, flip(aspace))
    @tensor F₊[a; c d] := Fdagup'[a, b, c] * iso'[d, b]
    @tensor Fdag₊[d a; c] := Fup'[a, b, c] * iso[b, d]
    F₊, Fdag₊
end
const FFdag₋1 = let
    Fdagdn, Fdn = FdagF₋1
    aspace = Rep[ℤ₂×U₁]((1, -1 // 2) => 1)
    iso = isometry(aspace, flip(aspace))
    @tensor F₋[a; c d] := Fdagdn'[a, b, c] * iso'[d, b]
    @tensor Fdag₋[d a; c] := Fdn'[a, b, c] * iso[b, d]
    F₋, Fdag₋
end

# zz hopping t⟂
const FdagzupFzdn₊ = let
    Fdag₊ = FdagF₊1[1]
    F₊ = FdagF₊1[2]
    @tensor FdagzupFzdn₊[p1; p2] := iso[p1, s1, s2, s3, s4] * Id1[s1, s1p] * Id1[s4, s4p] * Z1[s2, s2in] *
                                    Fdag₊[s2in, s2p, a] * F₊[a, s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    FdagzupFzdn₊
end

const FzupFdagzdn₊ = let
    F₊ = FFdag₊1[1]
    Fdag₊ = FFdag₊1[2]
    @tensor FzupFdagzdn₊[p1; p2] := iso[p1, s1, s2, s3, s4] * Id1[s1, s1p] * Id1[s4, s4p] * Z1[s2, s2in] *
                                    F₊[s2in, s2p, a] * Fdag₊[a, s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    FzupFdagzdn₊
end

const FdagzupFzdn₋ = let
    Fdag₋ = FdagF₋1[1]
    F₋ = FdagF₋1[2]
    @tensor FdagzupFzdn₋[p1; p2] := iso[p1, s1, s2, s3, s4] * Id1[s1, s1p] * Id1[s4, s4p] * Z1[s2, s2in] *
                                    Fdag₋[s2in, s2p, a] * F₋[a, s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    FdagzupFzdn₋
end

const FzupFdagzdn₋ = let
    F₋ = FFdag₋1[1]
    Fdag₋ = FFdag₋1[2]
    @tensor FzupFdagzdn₋[p1; p2] := iso[p1, s1, s2, s3, s4] * Id1[s1, s1p] * Id1[s4, s4p] * Z1[s2, s2in] *
                                    F₋[s2in, s2p, a] * Fdag₋[a, s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    FzupFdagzdn₋
end

# ====================== Nearest neighbor spin interaction =======================
const SxupSxup = let
    @tensor SxupSxup[(p1, p3); (p2, p4)] :=
        iso[p1, s1, s2, s3, s4] * iso'[s1p, s2, s3, s4, p2] * SS1[s1, ss1, s1p, ss1p] *
        iso[p3, ss1, ss2, ss3, ss4] * iso'[ss1p, ss2, ss3, ss4, p4]
    SxupSxup
end

const SxdnSxdn = let
    @tensor SxdnSxdn[(p1, p3); (p2, p4)] :=
        iso[p1, s1, s2, s3, s4] * iso'[s1, s2, s3, s4p, p2] * SS1[s4, ss4, s4p, ss4p] *
        iso[p3, ss1, ss2, ss3, ss4] * iso'[ss1, ss2, ss3, ss4p, p4]
    SxdnSxdn
end

const SzupSzup = let
    @tensor SzupSzup[(p1, p3); (p2, p4)] :=
        iso[p1, s1, s2, s3, s4] * iso'[s1, s2p, s3, s4, p2] * SS1[s2, ss2, s2p, ss2p] *
        iso[p3, ss1, ss2, ss3, ss4] * iso'[ss1, ss2p, ss3, ss4, p4]
    SzupSzup
end

const SzdnSzdn = let
    @tensor SzdnSzdn[(p1, p3); (p2, p4)] :=
        iso[p1, s1, s2, s3, s4] * iso'[s1, s2, s3p, s4, p2] * SS1[s3, ss3, s3p, ss3p] *
        iso[p3, ss1, ss2, ss3, ss4] * iso'[ss1, ss2, ss3p, ss4, p4]
    SzdnSzdn
end

const SxupSzup_intra = let
    @tensor SxupSzup_intra[(p1, p3); (p2, p4)] :=
        iso[p1, s1, s2, s3, s4] * iso'[s1p, s2, s3, s4, p2] * SS1[s1, ss2, s1p, ss2p] *
        iso[p3, ss1, ss2, ss3, ss4] * iso'[ss1, ss2p, ss3, ss4, p4]
    SxupSzup_intra
end

const SxdnSzdn_intra = let
    @tensor SxdnSzdn_intra[(p1, p3); (p2, p4)] :=
        iso[p1, s1, s2, s3, s4] * iso'[s1, s2, s3, s4p, p2] * SS1[s4, ss3, s4p, ss3p] *
        iso[p3, ss1, ss2, ss3, ss4] * iso'[ss1, ss2, ss3p, ss4, p4]
    SxdnSzdn_intra
end

# ====================== Nearest neighbor hopping term =======================
# ---------------- xx hopping ----------------
const FdagxupFxup₊ = let
    Fdag₊ = FdagF₊1[1]
    F₊ = FdagF₊1[2]
    @tensor Fdagxup₊[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * Fdag₊[s1, s1p, a] * iso'[s1p, s2, s3, s4, p2]
    @tensor Fxup₊[a, p1; p2] := iso[p1, s1, s2, s3, s4] * F₊[a, s1, s1p] * iso'[s1p, s2, s3, s4, p2]
    Fdagxup₊, Fxup₊
end

const FdagxupFxup₋ = let
    Fdag₋ = FdagF₋1[1]
    F₋ = FdagF₋1[2]
    @tensor Fdagxup₋[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * Fdag₋[s1, s1p, a] * iso'[s1p, s2, s3, s4, p2]
    @tensor Fxup₋[a, p1; p2] := iso[p1, s1, s2, s3, s4] * F₋[a, s1, s1p] * iso'[s1p, s2, s3, s4, p2]
    Fdagxup₋, Fxup₋
end

const FxupFdagxup₊ = let
    F₊ = FFdag₊1[1]
    Fdag₊ = FFdag₊1[2]
    @tensor Fxup₊[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * F₊[s1, s1p, a] * iso'[s1p, s2, s3, s4, p2]
    @tensor Fdagxup₊[a, p1; p2] := iso[p1, s1, s2, s3, s4] * Fdag₊[a, s1, s1p] * iso'[s1p, s2, s3, s4, p2]
    Fxup₊, Fdagxup₊
end

const FxupFdagxup₋ = let
    F₋ = FFdag₋1[1]
    Fdag₋ = FFdag₋1[2]
    @tensor Fxup₋[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * F₋[s1, s1p, a] * iso'[s1p, s2, s3, s4, p2]
    @tensor Fdagxup₋[a, p1; p2] := iso[p1, s1, s2, s3, s4] * Fdag₋[a, s1, s1p] * iso'[s1p, s2, s3, s4, p2]
    Fxup₋, Fdagxup₋
end

const FdagxdnFxdn₊ = let
    Fdag₊ = FdagF₊1[1]
    F₊ = FdagF₊1[2]
    @tensor Fdagxdn₊[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * Fdag₊[s4, s4p, a] * iso'[s1, s2, s3, s4p, p2]
    @tensor Fxdn₊[a, p1; p2] := iso[p1, s1, s2, s3, s4] * F₊[a, s4, s4p] * iso'[s1, s2, s3, s4p, p2]
    Fdagxdn₊, Fxdn₊
end

const FdagxdnFxdn₋ = let
    Fdag₋ = FdagF₋1[1]
    F₋ = FdagF₋1[2]
    @tensor Fdagxdn₋[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * Fdag₋[s4, s4p, a] * iso'[s1, s2, s3, s4p, p2]
    @tensor Fxdn₋[a, p1; p2] := iso[p1, s1, s2, s3, s4] * F₋[a, s4, s4p] * iso'[s1, s2, s3, s4p, p2]
    Fdagxdn₋, Fxdn₋
end

const FxdnFdagxdn₊ = let
    F₊ = FFdag₊1[1]
    Fdag₊ = FFdag₊1[2]
    @tensor Fxdn₊[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * F₊[s4, s4p, a] * iso'[s1, s2, s3, s4p, p2]
    @tensor Fdagxdn₊[a, p1; p2] := iso[p1, s1, s2, s3, s4] * Fdag₊[a, s4, s4p] * iso'[s1, s2, s3, s4p, p2]
    Fxdn₊, Fdagxdn₊
end

const FxdnFdagxdn₋ = let
    F₋ = FFdag₋1[1]
    Fdag₋ = FFdag₋1[2]
    @tensor Fxdn₋[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * F₋[s4, s4p, a] * iso'[s1, s2, s3, s4p, p2]
    @tensor Fdagxdn₋[a, p1; p2] := iso[p1, s1, s2, s3, s4] * Fdag₋[a, s4, s4p] * iso'[s1, s2, s3, s4p, p2]
    Fxdn₋, Fdagxdn₋
end

# ---------------- zz hopping ----------------
const FdagzupFzup₊ = let
    Fdag₊ = FdagF₊1[1]
    F₊ = FdagF₊1[2]
    @tensor Fdagzup₊[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * Fdag₊[s2, s2p, a] * iso'[s1, s2p, s3, s4, p2]
    @tensor Fzup₊[a, p1; p2] := iso[p1, s1, s2, s3, s4] * F₊[a, s2, s2p] * iso'[s1, s2p, s3, s4, p2]
    Fdagzup₊, Fzup₊
end

const FdagzupFzup₋ = let
    Fdag₋ = FdagF₋1[1]
    F₋ = FdagF₋1[2]
    @tensor Fdagzup₋[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * Fdag₋[s2, s2p, a] * iso'[s1, s2p, s3, s4, p2]
    @tensor Fzup₋[a, p1; p2] := iso[p1, s1, s2, s3, s4] * F₋[a, s2, s2p] * iso'[s1, s2p, s3, s4, p2]
    Fdagzup₋, Fzup₋
end

const FzupFdagzup₊ = let
    F₊ = FFdag₊1[1]
    Fdag₊ = FFdag₊1[2]
    @tensor Fzup₊[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * F₊[s2, s2p, a] * iso'[s1, s2p, s3, s4, p2]
    @tensor Fdagzup₊[a, p1; p2] := iso[p1, s1, s2, s3, s4] * Fdag₊[a, s2, s2p] * iso'[s1, s2p, s3, s4, p2]
    Fzup₊, Fdagzup₊
end

const FzupFdagzup₋ = let
    F₋ = FFdag₋1[1]
    Fdag₋ = FFdag₋1[2]
    @tensor Fzup₋[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * F₋[s2, s2p, a] * iso'[s1, s2p, s3, s4, p2]
    @tensor Fdagzup₋[a, p1; p2] := iso[p1, s1, s2, s3, s4] * Fdag₋[a, s2, s2p] * iso'[s1, s2p, s3, s4, p2]
    Fzup₋, Fdagzup₋
end

const FdagzdnFzdn₊ = let
    Fdag₊ = FdagF₊1[1]
    F₊ = FdagF₊1[2]
    @tensor Fdagzdn₊[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * Fdag₊[s3, s3p, a] * iso'[s1, s2, s3p, s4, p2]
    @tensor Fzdn₊[a, p1; p2] := iso[p1, s1, s2, s3, s4] * F₊[a, s3, s3p] * iso'[s1, s2, s3p, s4, p2]
    Fdagzdn₊, Fzdn₊
end

const FdagzdnFzdn₋ = let
    Fdag₋ = FdagF₋1[1]
    F₋ = FdagF₋1[2]
    @tensor Fdagzdn₋[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * Fdag₋[s3, s3p, a] * iso'[s1, s2, s3p, s4, p2]
    @tensor Fzdn₋[a, p1; p2] := iso[p1, s1, s2, s3, s4] * F₋[a, s3, s3p] * iso'[s1, s2, s3p, s4, p2]
    Fdagzdn₋, Fzdn₋
end

const FzdnFdagzdn₊ = let
    F₊ = FFdag₊1[1]
    Fdag₊ = FFdag₊1[2]
    @tensor Fxdn₊[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * F₊[s3, s3p, a] * iso'[s1, s2, s3p, s4, p2]
    @tensor Fdagxdn₊[a, p1; p2] := iso[p1, s1, s2, s3, s4] * Fdag₊[a, s3, s3p] * iso'[s1, s2, s3p, s4, p2]
    Fxdn₊, Fdagxdn₊
end

const FzdnFdagzdn₋ = let
    F₋ = FFdag₋1[1]
    Fdag₋ = FFdag₋1[2]
    @tensor Fxdn₋[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * F₋[s3, s3p, a] * iso'[s1, s2, s3p, s4, p2]
    @tensor Fdagxdn₋[a, p1; p2] := iso[p1, s1, s2, s3, s4] * Fdag₋[a, s3, s3p] * iso'[s1, s2, s3p, s4, p2]
    Fxdn₋, Fdagxdn₋
end

# ---------------- xz hopping ----------------
const FdagxupFzup₊ = let
    Fdag₊ = FdagF₊1[1]
    F₊ = FdagF₊1[2]
    @tensor Fdagxup₊[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * Fdag₊[s1, s1p, a] * iso'[s1p, s2, s3, s4, p2]
    @tensor Fzup₊[a, p1; p2] := iso[p1, s1, s2, s3, s4] * F₊[a, s2, s2p] * iso'[s1, s2p, s3, s4, p2]
    Fdagxup₊, Fzup₊
end

const FdagxupFzup₋ = let
    Fdag₋ = FdagF₋1[1]
    F₋ = FdagF₋1[2]
    @tensor Fdagxup₋[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * Fdag₋[s1, s1p, a] * iso'[s1p, s2, s3, s4, p2]
    @tensor Fzup₋[a, p1; p2] := iso[p1, s1, s2, s3, s4] * F₋[a, s2, s2p] * iso'[s1, s2p, s3, s4, p2]
    Fdagxup₋, Fzup₋
end

const FxupFdagzup₊ = let
    F₊ = FFdag₊1[1]
    Fdag₊ = FFdag₊1[2]
    @tensor Fxup₊[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * F₊[s1, s1p, a] * iso'[s1p, s2, s3, s4, p2]
    @tensor Fdagzup₊[a, p1; p2] := iso[p1, s1, s2, s3, s4] * Fdag₊[a, s2, s2p] * iso'[s1, s2p, s3, s4, p2]
    Fxup₊, Fdagzup₊
end

const FxupFdagzup₋ = let
    F₋ = FFdag₋1[1]
    Fdag₋ = FFdag₋1[2]
    @tensor Fxup₋[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * F₋[s1, s1p, a] * iso'[s1p, s2, s3, s4, p2]
    @tensor Fdagzup₋[a, p1; p2] := iso[p1, s1, s2, s3, s4] * Fdag₋[a, s2, s2p] * iso'[s1, s2p, s3, s4, p2]
    Fxup₋, Fdagzup₋
end

const FdagzupFxup₊ = let
    Fdag₊ = FdagF₊1[1]
    F₊ = FdagF₊1[2]
    @tensor Fdagzup₊[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * Fdag₊[s2, s2p, a] * iso'[s1, s2p, s3, s4, p2]
    @tensor Fxup₊[a, p1; p2] := iso[p1, s1, s2, s3, s4] * F₊[a, s1, s1p] * iso'[s1p, s2, s3, s4, p2]
    Fdagzup₊, Fxup₊
end

const FdagzupFxup₋ = let
    Fdag₋ = FdagF₋1[1]
    F₋ = FdagF₋1[2]
    @tensor Fdagzup₋[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * Fdag₋[s2, s2p, a] * iso'[s1, s2p, s3, s4, p2]
    @tensor Fxup₋[a, p1; p2] := iso[p1, s1, s2, s3, s4] * F₋[a, s1, s1p] * iso'[s1p, s2, s3, s4, p2]
    Fdagzup₋, Fxup₋
end

const FzupFdagxup₊ = let
    F₊ = FFdag₊1[1]
    Fdag₊ = FFdag₊1[2]
    @tensor Fzup₊[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * F₊[s2, s2p, a] * iso'[s1, s2p, s3, s4, p2]
    @tensor Fdagxup₊[a, p1; p2] := iso[p1, s1, s2, s3, s4] * Fdag₊[a, s1, s1p] * iso'[s1p, s2, s3, s4, p2]
    Fzup₊, Fdagxup₊
end

const FzupFdagxup₋ = let
    F₋ = FFdag₋1[1]
    Fdag₋ = FFdag₋1[2]
    @tensor Fzup₋[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * F₋[s2, s2p, a] * iso'[s1, s2p, s3, s4, p2]
    @tensor Fdagxup₋[a, p1; p2] := iso[p1, s1, s2, s3, s4] * Fdag₋[a, s1, s1p] * iso'[s1p, s2, s3, s4, p2]
    Fzup₋, Fdagxup₋
end

const FdagxdnFzdn₊ = let
    Fdag₊ = FdagF₊1[1]
    F₊ = FdagF₊1[2]
    @tensor Fdagxdn₊[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * Fdag₊[s4, s4p, a] * iso'[s1, s2, s3, s4p, p2]
    @tensor Fzdn₊[a, p1; p2] := iso[p1, s1, s2, s3, s4] * F₊[a, s3, s3p] * iso'[s1, s2, s3p, s4, p2]
    Fdagxdn₊, Fzdn₊
end

const FdagxdnFzdn₋ = let
    Fdag₋ = FdagF₋1[1]
    F₋ = FdagF₋1[2]
    @tensor Fdagxdn₋[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * Fdag₋[s4, s4p, a] * iso'[s1, s2, s3, s4p, p2]
    @tensor Fzdn₋[a, p1; p2] := iso[p1, s1, s2, s3, s4] * F₋[a, s3, s3p] * iso'[s1, s2, s3p, s4, p2]
    Fdagxdn₋, Fzdn₋
end

const FxdnFdagzdn₊ = let
    F₊ = FFdag₊1[1]
    Fdag₊ = FFdag₊1[2]
    @tensor Fxdn₊[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * F₊[s4, s4p, a] * iso'[s1, s2, s3, s4p, p2]
    @tensor Fdagzdn₊[a, p1; p2] := iso[p1, s1, s2, s3, s4] * Fdag₊[a, s3, s3p] * iso'[s1, s2, s3p, s4, p2]
    Fxdn₊, Fdagzdn₊
end

const FxdnFdagzdn₋ = let
    F₋ = FFdag₋1[1]
    Fdag₋ = FFdag₋1[2]
    @tensor Fxdn₋[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * F₋[s4, s4p, a] * iso'[s1, s2, s3, s4p, p2]
    @tensor Fdagzdn₋[a, p1; p2] := iso[p1, s1, s2, s3, s4] * Fdag₋[a, s3, s3p] * iso'[s1, s2, s3p, s4, p2]
    Fxdn₋, Fdagzdn₋
end

const FdagzdnFxdn₊ = let
    Fdag₊ = FdagF₊1[1]
    F₊ = FdagF₊1[2]
    @tensor Fdagzdn₊[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * Fdag₊[s3, s3p, a] * iso'[s1, s2, s3p, s4, p2]
    @tensor Fxdn₊[a, p1; p2] := iso[p1, s1, s2, s3, s4] * F₊[a, s4, s4p] * iso'[s1, s2, s3, s4p, p2]
    Fdagzdn₊, Fxdn₊
end

const FdagzdnFxdn₋ = let
    Fdag₋ = FdagF₋1[1]
    F₋ = FdagF₋1[2]
    @tensor Fdagzdn₋[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * Fdag₋[s3, s3p, a] * iso'[s1, s2, s3p, s4, p2]
    @tensor Fxdn₋[a, p1; p2] := iso[p1, s1, s2, s3, s4] * F₋[a, s4, s4p] * iso'[s1, s2, s3, s4p, p2]
    Fdagzdn₋, Fxdn₋
end

const FzdnFdagxdn₊ = let
    F₊ = FFdag₊1[1]
    Fdag₊ = FFdag₊1[2]
    @tensor Fzdn₊[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * F₊[s3, s3p, a] * iso'[s1, s2, s3p, s4, p2]
    @tensor Fdagxdn₊[a, p1; p2] := iso[p1, s1, s2, s3, s4] * Fdag₊[a, s4, s4p] * iso'[s1, s2, s3, s4p, p2]
    Fzdn₊, Fdagxdn₊
end

const FzdnFdagxdn₋ = let
    F₋ = FFdag₋1[1]
    Fdag₋ = FFdag₋1[2]
    @tensor Fzdn₋[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * F₋[s3, s3p, a] * iso'[s1, s2, s3p, s4, p2]
    @tensor Fdagxdn₋[a, p1; p2] := iso[p1, s1, s2, s3, s4] * Fdag₋[a, s4, s4p] * iso'[s1, s2, s3, s4p, p2]
    Fzdn₋, Fdagxdn₋
end

# TODO: continue my work here.
# ---------------------- interlayer onsite pairing ---------------------
# singlet Δₛ = (c↑c↓ - c↓c↑)/√2  (onsite terms)
const Δₛdagx = let
    A = FdagF₋1[1]
    B = FFdag₊1[2]
    C = FdagF₊1[1]
    D = FFdag₋1[2]
    @tensor dnup[p1; p2] := iso[p1, s1, s2, s3, s4] * Z1[s1, s1in] * A[s1in, s1p, a] * B[a, s4, s4p] *
                            Z1[s2, s2p] * Z1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor updn[p1; p2] := iso[p1, s1, s2, s3, s4] * Z1[s1, s1in] * C[s1in, s1p, a] * D[a, s4, s4p] *
                            Z1[s2, s2p] * Z1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    (dnup - updn) / sqrt(2)
end

const Δₛx = let
    A = FFdag₊1[1]
    B = FdagF₋1[2]
    C = FFdag₋1[1]
    D = FdagF₊1[2]
    @tensor updn[p1; p2] := iso[p1, s1, s2, s3, s4] * Z1[s1, s1in] * A[s1in, s1p, a] * B[a, s4, s4p] *
                            Z1[s2, s2p] * Z1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor dnup[p1; p2] := iso[p1, s1, s2, s3, s4] * Z1[s1, s1in] * C[s1in, s1p, a] * D[a, s4, s4p] *
                            Z1[s2, s2p] * Z1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
    (updn - dnup) / sqrt(2)
end

const Δₛdagz = let
    A = FdagF₋1[1]
    B = FFdag₊1[2]
    C = FdagF₊1[1]
    D = FFdag₋1[2]
    @tensor dnup[p1; p2] := iso[p1, s1, s2, s3, s4] * Z1[s2, s2in] * A[s2in, s2p, a] * B[a, s3, s3p] *
                            Id1[s1, s1p] * Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor updn[p1; p2] := iso[p1, s1, s2, s3, s4] * Z1[s2, s2in] * C[s2in, s2p, a] * D[a, s3, s3p] *
                            Id1[s1, s1p] * Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    (dnup - updn) / sqrt(2)
end

const Δₛz = let
    A = FFdag₊1[1]
    B = FdagF₋1[2]
    C = FFdag₋1[1]
    D = FdagF₊1[2]
    @tensor updn[p1; p2] := iso[p1, s1, s2, s3, s4] * Z1[s2, s2in] * A[s2in, s2p, a] * B[a, s3, s3p] *
                            Id1[s1, s1p] * Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor dnup[p1; p2] := iso[p1, s1, s2, s3, s4] * Z1[s2, s2in] * C[s2in, s2p, a] * D[a, s3, s3p] *
                            Id1[s1, s1p] * Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    (updn - dnup) / sqrt(2)
end

const Δₛdagxz = let
    A = FdagF₋1[1]
    B = FFdag₊1[2]
    C = FdagF₊1[1]
    D = FFdag₋1[2]
    @tensor dnup[p1; p2] := iso[p1, s1, s2, s3, s4] * Z1[s1, s1in] * A[s1in, s1p, a] * B[a, s3, s3p] *
                            Z1[s2, s2p] * Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor updn[p1; p2] := iso[p1, s1, s2, s3, s4] * Z1[s1, s1in] * C[s1in, s1p, a] * D[a, s3, s3p] *
                            Z1[s2, s2p] * Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    (dnup - updn) / sqrt(2)
end

const Δₛxz = let
    A = FFdag₊1[1]
    B = FdagF₋1[2]
    C = FFdag₋1[1]
    D = FdagF₊1[2]
    @tensor updn[p1; p2] := iso[p1, s1, s2, s3, s4] * Z1[s1, s1in] * A[s1in, s1p, a] * B[a, s3, s3p] *
                            Z1[s2, s2p] * Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor dnup[p1; p2] := iso[p1, s1, s2, s3, s4] * Z1[s1, s1in] * C[s1in, s1p, a] * D[a, s3, s3p] *
                            Z1[s2, s2p] * Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
    (updn - dnup) / sqrt(2)
end

const Δₛdagzx = let
    A = FdagF₋1[1]
    B = FFdag₊1[2]
    C = FdagF₊1[1]
    D = FFdag₋1[2]
    @tensor dnup[p1; p2] := iso[p1, s1, s2, s3, s4] * Z1[s2, s2in] * A[s2in, s2p, a] * B[a, s4, s4p] *
                            Z1[s3, s3p] * Id1[s1, s1p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor updn[p1; p2] := iso[p1, s1, s2, s3, s4] * Z1[s2, s2in] * C[s2in, s2p, a] * D[a, s4, s4p] *
                            Z1[s3, s3p] * Id1[s1, s1p] * iso'[s1p, s2p, s3p, s4p, p2]
    (dnup - updn) / sqrt(2)
end

const Δₛzx = let
    A = FFdag₊1[1]
    B = FdagF₋1[2]
    C = FFdag₋1[1]
    D = FdagF₊1[2]
    @tensor updn[p1; p2] := iso[p1, s1, s2, s3, s4] * Z1[s2, s2in] * A[s2in, s2p, a] * B[a, s4, s4p] *
                            Z1[s3, s3p] * Id1[s1, s1p] * iso'[s1p, s2p, s3p, s4p, p2]
    @tensor dnup[p1; p2] := iso[p1, s1, s2, s3, s4] * Z1[s2, s2in] * C[s2in, s2p, a] * D[a, s4, s4p] *
                            Z1[s3, s3p] * Id1[s1, s1p] * iso'[s1p, s2p, s3p, s4p, p2]
    (updn - dnup) / sqrt(2)
end

# ---------------------- interlayer 2site pairing ---------------------
# const Δₛdagx2site = let
#     A = FdagF1[1]
#     B = FFdag1[2]
#     @tensor Fdagxup[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * A[s1, s1p, a] * Id1[s4, s4p] * Id1[s2, s2p] *
#                                     Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
#     @tensor Fdagxdn[a, p1; p2] := iso[p1, s1, s2, s3, s4] * B[a, s4, s4p] * Id1[s1, s1p] * Id1[s2, s2p] *
#                                   Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
#     @tensor Δₛdagx2site[p1, p3; p2, p4] := Zxup[p1, p1in] * Fdagxup[p1in, p2, a] * Fdagxdn[a, p3, p4]
#     Δₛdagx2site
# end

# const Δₛx2site = let
#     A = FFdag1[1]
#     B = FdagF1[2]
#     @tensor Fxup[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * A[s1, s1p, a] * Id1[s4, s4p] * Id1[s2, s2p] *
#                                  Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
#     @tensor Fxdn[a, p1; p2] := iso[p1, s1, s2, s3, s4] * B[a, s4, s4p] * Id1[s1, s1p] * Id1[s2, s2p] *
#                                Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
#     @tensor Δₛx2site[p1, p3; p2, p4] := Zxup[p1, p1in] * Fxup[p1in, p2, a] * Fxdn[a, p3, p4]
#     Δₛx2site
# end

# const Δₛdagz2site = let
#     A = FdagF1[1]
#     B = FFdag1[2]
#     @tensor Fdagzup[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * A[s2, s2p, a] * Id1[s4, s4p] * Id1[s1, s1p] *
#                                     Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
#     @tensor Fdagzdn[a, p1; p2] := iso[p1, s1, s2, s3, s4] * B[a, s3, s3p] * Id1[s1, s1p] * Id1[s2, s2p] *
#                                   Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
#     @tensor Δₛdagz2site[p1, p3; p2, p4] := Zzup[p1, p1in] * Fdagzup[p1in, p2, a] * Fdagzdn[a, p3, p4]
#     Δₛdagz2site
# end

# const Δₛz2site = let
#     A = FFdag1[1]
#     B = FdagF1[2]
#     @tensor Fzup[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * A[s2, s2p, a] * Id1[s4, s4p] * Id1[s1, s1p] *
#                                  Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
#     @tensor Fzdn[a, p1; p2] := iso[p1, s1, s2, s3, s4] * B[a, s3, s3p] * Id1[s1, s1p] * Id1[s2, s2p] *
#                                Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
#     @tensor Δₛz2site[p1, p3; p2, p4] := Zzup[p1, p1in] * Fzup[p1in, p2, a] * Fzdn[a, p3, p4]
#     Δₛz2site
# end

# const Δₛdagxz2site = let
#     A = FdagF1[1]
#     B = FFdag1[2]
#     @tensor Fdagxup[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * A[s1, s1p, a] * Id1[s4, s4p] * Id1[s2, s2p] *
#                                     Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
#     @tensor Fdagzdn[a, p1; p2] := iso[p1, s1, s2, s3, s4] * B[a, s3, s3p] * Id1[s1, s1p] * Id1[s2, s2p] *
#                                   Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
#     @tensor Δₛdagxz2site[p1, p3; p2, p4] := Zxup[p1, p1in] * Fdagxup[p1in, p2, a] * Fdagzdn[a, p3, p4]
#     Δₛdagxz2site
# end

# const Δₛxz2site = let
#     A = FFdag1[1]
#     B = FdagF1[2]
#     @tensor Fxup[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * A[s1, s1p, a] * Id1[s4, s4p] * Id1[s2, s2p] *
#                                  Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
#     @tensor Fzdn[a, p1; p2] := iso[p1, s1, s2, s3, s4] * B[a, s3, s3p] * Id1[s1, s1p] * Id1[s2, s2p] *
#                                Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
#     @tensor Δₛxz2site[p1, p3; p2, p4] := Zxup[p1, p1in] * Fxup[p1in, p2, a] * Fzdn[a, p3, p4]
#     Δₛxz2site
# end

# const Δₛdagzx2site = let
#     A = FdagF1[1]
#     B = FFdag1[2]
#     @tensor Fdagzup[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * A[s2, s2p, a] * Id1[s4, s4p] * Id1[s1, s1p] *
#                                     Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
#     @tensor Fdagxdn[a, p1; p2] := iso[p1, s1, s2, s3, s4] * B[a, s4, s4p] * Id1[s1, s1p] * Id1[s2, s2p] *
#                                   Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
#     @tensor Δₛdagzx2site[p1, p3; p2, p4] := Zzup[p1, p1in] * Fdagzup[p1in, p2, a] * Fdagxdn[a, p3, p4]
#     Δₛdagzx2site
# end

# const Δₛzx2site = let
#     A = FFdag1[1]
#     B = FdagF1[2]
#     @tensor Fzup[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * A[s2, s2p, a] * Id1[s4, s4p] * Id1[s1, s1p] *
#                                  Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
#     @tensor Fxdn[a, p1; p2] := iso[p1, s1, s2, s3, s4] * B[a, s4, s4p] * Id1[s1, s1p] * Id1[s2, s2p] *
#                                Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
#     @tensor Δₛzx2site[p1, p3; p2, p4] := Zzup[p1, p1in] * Fzup[p1in, p2, a] * Fxdn[a, p3, p4]
#     Δₛzx2site
# end

# ---------------------- intralayer 2-site pairing ---------------------
# const Δₛdagxup_intra = let
#     A = FdagF1[1]
#     B = FFdag1[2]
#     @tensor Fdagxup[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * A[s1, s1p, a] * Id1[s4, s4p] * Id1[s2, s2p] *
#                                     Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
#     @tensor Fdagxup2[a, p1; p2] := iso[p1, s1, s2, s3, s4] * B[a, s1, s1p] * Id1[s4, s4p] * Id1[s2, s2p] *
#                                    Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
#     @tensor Δₛdagxup_intra[p1, p3; p2, p4] := Zxup[p1, p1in] * Fdagxup[p1in, p2, a] * Fdagxup2[a, p3, p4]
#     Δₛdagxup_intra
# end

# const Δₛdagxdn_intra = let
#     A = FdagF1[1]
#     B = FFdag1[2]
#     @tensor Fdagxdn[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * A[s4, s4p, a] * Id1[s1, s1p] * Id1[s2, s2p] *
#                                     Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
#     @tensor Fdagxdn2[a, p1; p2] := iso[p1, s1, s2, s3, s4] * B[a, s4, s4p] * Id1[s1, s1p] * Id1[s2, s2p] *
#                                    Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
#     @tensor Δₛdagxdn_intra[p1, p3; p2, p4] := Zxdn[p1, p1in] * Fdagxdn[p1in, p2, a] * Fdagxdn2[a, p3, p4]
#     Δₛdagxdn_intra
# end

# const Δₛxup_intra = let
#     A = FFdag1[1]
#     B = FdagF1[2]
#     @tensor Fxup[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * A[s1, s1p, a] * Id1[s4, s4p] * Id1[s2, s2p] *
#                                  Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
#     @tensor Fxup2[a, p1; p2] := iso[p1, s1, s2, s3, s4] * B[a, s1, s1p] * Id1[s4, s4p] * Id1[s2, s2p] *
#                                 Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
#     @tensor Δₛxup_intra[p1, p3; p2, p4] := Zxup[p1, p1in] * Fxup[p1in, p2, a] * Fxup2[a, p3, p4]
#     Δₛxup_intra
# end

# const Δₛxdn_intra = let
#     A = FFdag1[1]
#     B = FdagF1[2]
#     @tensor Fxdn[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * A[s4, s4p, a] * Id1[s1, s1p] * Id1[s2, s2p] *
#                                  Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
#     @tensor Fxdn2[a, p1; p2] := iso[p1, s1, s2, s3, s4] * B[a, s4, s4p] * Id1[s1, s1p] * Id1[s2, s2p] *
#                                 Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
#     @tensor Δₛxdn_intra[p1, p3; p2, p4] := Zxdn[p1, p1in] * Fxdn[p1in, p2, a] * Fxdn2[a, p3, p4]
#     Δₛxdn_intra
# end

# const Δₛdagxzup_intra = let
#     A = FdagF1[1]
#     B = FFdag1[2]
#     @tensor Fdagxup[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * A[s1, s1p, a] * Id1[s4, s4p] * Id1[s2, s2p] *
#                                     Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
#     @tensor Fdagzup[a, p1; p2] := iso[p1, s1, s2, s3, s4] * B[a, s2, s2p] * Id1[s4, s4p] * Id1[s1, s1p] *
#                                   Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
#     @tensor Δₛdagxzup_intra[p1, p3; p2, p4] := Zxup[p1, p1in] * Fdagxup[p1in, p2, a] * Fdagzup[a, p3, p4]
#     Δₛdagxzup_intra
# end

# const Δₛdagxzdn_intra = let
#     A = FdagF1[1]
#     B = FFdag1[2]
#     @tensor Fdagxdn[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * A[s4, s4p, a] * Id1[s1, s1p] * Id1[s2, s2p] *
#                                     Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
#     @tensor Fdagzdn[a, p1; p2] := iso[p1, s1, s2, s3, s4] * B[a, s3, s3p] * Id1[s1, s1p] * Id1[s2, s2p] *
#                                   Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
#     @tensor Δₛdagxzdn_intra[p1, p3; p2, p4] := Zxdn[p1, p1in] * Fdagxdn[p1in, p2, a] * Fdagzdn[a, p3, p4]
#     Δₛdagxzdn_intra
# end

# const Δₛxzup_intra = let
#     A = FFdag1[1]
#     B = FdagF1[2]
#     @tensor Fxup[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * A[s1, s1p, a] * Id1[s4, s4p] * Id1[s2, s2p] *
#                                  Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
#     @tensor Fzup[a, p1; p2] := iso[p1, s1, s2, s3, s4] * B[a, s2, s2p] * Id1[s4, s4p] * Id1[s1, s1p] *
#                                Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
#     @tensor Δₛxzup_intra[p1, p3; p2, p4] := Zxup[p1, p1in] * Fxup[p1in, p2, a] * Fzup[a, p3, p4]
#     Δₛxzup_intra
# end

# const Δₛxzdn_intra = let
#     A = FFdag1[1]
#     B = FdagF1[2]
#     @tensor Fxdn[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * A[s4, s4p, a] * Id1[s1, s1p] * Id1[s2, s2p] *
#                                  Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
#     @tensor Fzdn[a, p1; p2] := iso[p1, s1, s2, s3, s4] * B[a, s3, s3p] * Id1[s1, s1p] * Id1[s2, s2p] *
#                                Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
#     @tensor Δₛxzdn_intra[p1, p3; p2, p4] := Zxdn[p1, p1in] * Fxdn[p1in, p2, a] * Fzdn[a, p3, p4]
#     Δₛxzdn_intra
# end

# const Δₛdagzup_intra = let
#     A = FdagF1[1]
#     B = FFdag1[2]
#     @tensor Fdagzup[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * A[s2, s2p, a] * Id1[s4, s4p] * Id1[s1, s1p] *
#                                     Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
#     @tensor Fdagzup2[a, p1; p2] := iso[p1, s1, s2, s3, s4] * B[a, s2, s2p] * Id1[s4, s4p] * Id1[s1, s1p] *
#                                    Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
#     @tensor Δₛdagzup_intra[p1, p3; p2, p4] := Zzup[p1, p1in] * Fdagzup[p1in, p2, a] * Fdagzup2[a, p3, p4]
#     Δₛdagzup_intra
# end

# const Δₛdagzdn_intra = let
#     A = FdagF1[1]
#     B = FFdag1[2]
#     @tensor Fdagzdn[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * A[s3, s3p, a] * Id1[s1, s1p] * Id1[s2, s2p] *
#                                     Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
#     @tensor Fdagzdn2[a, p1; p2] := iso[p1, s1, s2, s3, s4] * B[a, s3, s3p] * Id1[s1, s1p] * Id1[s2, s2p] *
#                                    Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
#     @tensor Δₛdagzdn_intra[p1, p3; p2, p4] := Zzdn[p1, p1in] * Fdagzdn[p1in, p2, a] * Fdagzdn2[a, p3, p4]
#     Δₛdagzdn_intra
# end

# const Δₛzup_intra = let
#     A = FFdag1[1]
#     B = FdagF1[2]
#     @tensor Fzup[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * A[s2, s2p, a] * Id1[s4, s4p] * Id1[s1, s1p] *
#                                  Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
#     @tensor Fzup2[a, p1; p2] := iso[p1, s1, s2, s3, s4] * B[a, s2, s2p] * Id1[s4, s4p] * Id1[s1, s1p] *
#                                 Id1[s3, s3p] * iso'[s1p, s2p, s3p, s4p, p2]
#     @tensor Δₛzup_intra[p1, p3; p2, p4] := Zzup[p1, p1in] * Fzup[p1in, p2, a] * Fzup2[a, p3, p4]
#     Δₛzup_intra
# end

# const Δₛzdn_intra = let
#     A = FFdag1[1]
#     B = FdagF1[2]
#     @tensor Fzdn[p1; (p2, a)] := iso[p1, s1, s2, s3, s4] * A[s3, s3p, a] * Id1[s1, s1p] * Id1[s2, s2p] *
#                                  Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
#     @tensor Fzdn2[a, p1; p2] := iso[p1, s1, s2, s3, s4] * B[a, s3, s3p] * Id1[s1, s1p] * Id1[s2, s2p] *
#                                 Id1[s4, s4p] * iso'[s1p, s2p, s3p, s4p, p2]
#     @tensor Δₛzdn_intra[p1, p3; p2, p4] := Zzdn[p1, p1in] * Fzdn[p1in, p2, a] * Fzdn2[a, p3, p4]
#     Δₛzdn_intra
# end

end

"""
     const Z2U1tJ2orb = Z₂U₁tJ2orb
"""
const Z2U1tJ2orb = Z₂U₁tJ2orb


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

    Zxup = Z2U1tJ2orb.Zxup
    Zxdn = Z2U1tJ2orb.Zxdn
    Zzup = Z2U1tJ2orb.Zzup
    Zzdn = Z2U1tJ2orb.Zzdn
    # intralayer xx hopping
    Fdagxup₊, Fxup₊ = Z2U1tJ2orb.FdagxupFxup₊
    @tensor fdagfxup₊[p1, p3; p2, p4] := Zxup[p1, p1in] * Fdagxup₊[p1in, p2, a] * Fxup₊[a, p3, p4]
    Fxup₊, Fdagxup₊ = Z2U1tJ2orb.FxupFdagxup₊
    @tensor ffdagxup₊[p1, p3; p2, p4] := Zxup[p1, p1in] * Fxup₊[p1in, p2, a] * Fdagxup₊[a, p3, p4]
    Fdagxup₋, Fxup₋ = Z2U1tJ2orb.FdagxupFxup₋
    @tensor fdagfxup₋[p1, p3; p2, p4] := Zxup[p1, p1in] * Fdagxup₋[p1in, p2, a] * Fxup₋[a, p3, p4]
    Fxup₋, Fdagxup₋ = Z2U1tJ2orb.FxupFdagxup₋
    @tensor ffdagxup₋[p1, p3; p2, p4] := Zxup[p1, p1in] * Fxup₋[p1in, p2, a] * Fdagxup₋[a, p3, p4]

    Fdagxdn₊, Fxdn₊ = Z2U1tJ2orb.FdagxdnFxdn₊
    @tensor fdagfxdn₊[p1, p3; p2, p4] := Zxdn[p1, p1in] * Fdagxdn₊[p1in, p2, a] * Fxdn₊[a, p3, p4]
    Fxdn₊, Fdagxdn₊ = Z2U1tJ2orb.FxdnFdagxdn₊
    @tensor ffdagxdn₊[p1, p3; p2, p4] := Zxdn[p1, p1in] * Fxdn₊[p1in, p2, a] * Fdagxdn₊[a, p3, p4]
    Fdagxdn₋, Fxdn₋ = Z2U1tJ2orb.FdagxdnFxdn₋
    @tensor fdagfxdn₋[p1, p3; p2, p4] := Zxdn[p1, p1in] * Fdagxdn₋[p1in, p2, a] * Fxdn₋[a, p3, p4]
    Fxdn₋, Fdagxdn₋ = Z2U1tJ2orb.FxdnFdagxdn₋
    @tensor ffdagxdn₋[p1, p3; p2, p4] := Zxdn[p1, p1in] * Fxdn₋[p1in, p2, a] * Fdagxdn₋[a, p3, p4]

    # intralayer zz hopping
    Fdagzup₊, Fzup₊ = Z2U1tJ2orb.FdagzupFzup₊
    @tensor fdagfzup₊[p1, p3; p2, p4] := Zzup[p1, p1in] * Fdagzup₊[p1in, p2, a] * Fzup₊[a, p3, p4]
    Fzup₊, Fdagzup₊ = Z2U1tJ2orb.FzupFdagzup₊
    @tensor ffdagzup₊[p1, p3; p2, p4] := Zzup[p1, p1in] * Fzup₊[p1in, p2, a] * Fdagzup₊[a, p3, p4]
    Fdagzup₋, Fzup₋ = Z2U1tJ2orb.FdagzupFzup₋
    @tensor fdagfzup₋[p1, p3; p2, p4] := Zzup[p1, p1in] * Fdagzup₋[p1in, p2, a] * Fzup₋[a, p3, p4]
    Fzup₋, Fdagzup₋ = Z2U1tJ2orb.FzupFdagzup₋
    @tensor ffdagzup₋[p1, p3; p2, p4] := Zzup[p1, p1in] * Fzup₋[p1in, p2, a] * Fdagzup₋[a, p3, p4]

    Fdagzdn₊, Fzdn₊ = Z2U1tJ2orb.FdagzdnFzdn₊
    @tensor fdagfzdn₊[p1, p3; p2, p4] := Zzdn[p1, p1in] * Fdagzdn₊[p1in, p2, a] * Fzdn₊[a, p3, p4]
    Fzdn₊, Fdagzdn₊ = Z2U1tJ2orb.FzdnFdagzdn₊
    @tensor ffdagzdn₊[p1, p3; p2, p4] := Zzdn[p1, p1in] * Fzdn₊[p1in, p2, a] * Fdagzdn₊[a, p3, p4]
    Fdagzdn₋, Fzdn₋ = Z2U1tJ2orb.FdagzdnFzdn₋
    @tensor fdagfzdn₋[p1, p3; p2, p4] := Zzdn[p1, p1in] * Fdagzdn₋[p1in, p2, a] * Fzdn₋[a, p3, p4]
    Fzdn₋, Fdagzdn₋ = Z2U1tJ2orb.FzdnFdagzdn₋
    @tensor ffdagzdn₋[p1, p3; p2, p4] := Zzdn[p1, p1in] * Fzdn₋[p1in, p2, a] * Fdagzdn₋[a, p3, p4]

    # intralayer xz hopping
    Fdagxup₊, Fzup₊ = Z2U1tJ2orb.FdagxupFzup₊
    @tensor fdagfxzup₊[p1, p3; p2, p4] := Zxup[p1, p1in] * Fdagxup₊[p1in, p2, a] * Fzup₊[a, p3, p4]
    Fxup₊, Fdagzup₊ = Z2U1tJ2orb.FxupFdagzup₊
    @tensor ffdagxzup₊[p1, p3; p2, p4] := Zxup[p1, p1in] * Fxup₊[p1in, p2, a] * Fdagzup₊[a, p3, p4]
    Fdagxup₋, Fzup₋ = Z2U1tJ2orb.FdagxupFzup₋
    @tensor fdagfxzup₋[p1, p3; p2, p4] := Zxup[p1, p1in] * Fdagxup₋[p1in, p2, a] * Fzup₋[a, p3, p4]
    Fxup₋, Fdagzup₋ = Z2U1tJ2orb.FxupFdagzup₋
    @tensor ffdagxzup₋[p1, p3; p2, p4] := Zxup[p1, p1in] * Fxup₋[p1in, p2, a] * Fdagzup₋[a, p3, p4]

    Fdagzup₊, Fxup₊ = Z2U1tJ2orb.FdagzupFxup₊
    @tensor fdagfzxup₊[p1, p3; p2, p4] := Zzup[p1, p1in] * Fdagzup₊[p1in, p2, a] * Fxup₊[a, p3, p4]
    Fzup₊, Fdagxup₊ = Z2U1tJ2orb.FzupFdagxup₊
    @tensor ffdagzxup₊[p1, p3; p2, p4] := Zzup[p1, p1in] * Fzup₊[p1in, p2, a] * Fdagxup₊[a, p3, p4]
    Fdagzup₋, Fxup₋ = Z2U1tJ2orb.FdagzupFxup₋
    @tensor fdagfzxup₋[p1, p3; p2, p4] := Zzup[p1, p1in] * Fdagzup₋[p1in, p2, a] * Fxup₋[a, p3, p4]
    Fzup₋, Fdagxup₋ = Z2U1tJ2orb.FzupFdagxup₋
    @tensor ffdagzxup₋[p1, p3; p2, p4] := Zzup[p1, p1in] * Fzup₋[p1in, p2, a] * Fdagxup₋[a, p3, p4]

    Fdagxdn₊, Fzdn₊ = Z2U1tJ2orb.FdagxdnFzdn₊
    @tensor fdagfxzdn₊[p1, p3; p2, p4] := Zxdn[p1, p1in] * Fdagxdn₊[p1in, p2, a] * Fzdn₊[a, p3, p4]
    Fxdn₊, Fdagzdn₊ = Z2U1tJ2orb.FxdnFdagzdn₊
    @tensor ffdagxzdn₊[p1, p3; p2, p4] := Zxdn[p1, p1in] * Fxdn₊[p1in, p2, a] * Fdagzdn₊[a, p3, p4]
    Fdagxdn₋, Fzdn₋ = Z2U1tJ2orb.FdagxdnFzdn₋
    @tensor fdagfxzdn₋[p1, p3; p2, p4] := Zxdn[p1, p1in] * Fdagxdn₋[p1in, p2, a] * Fzdn₋[a, p3, p4]
    Fxdn₋, Fdagzdn₋ = Z2U1tJ2orb.FxdnFdagzdn₋
    @tensor ffdagxzdn₋[p1, p3; p2, p4] := Zxdn[p1, p1in] * Fxdn₋[p1in, p2, a] * Fdagzdn₋[a, p3, p4]

    Fdagzdn₊, Fxdn₊ = Z2U1tJ2orb.FdagzdnFxdn₊
    @tensor fdagfzxdn₊[p1, p3; p2, p4] := Zzdn[p1, p1in] * Fdagzdn₊[p1in, p2, a] * Fxdn₊[a, p3, p4]
    Fzdn₊, Fdagxdn₊ = Z2U1tJ2orb.FzdnFdagxdn₊
    @tensor ffdagzxdn₊[p1, p3; p2, p4] := Zzdn[p1, p1in] * Fzdn₊[p1in, p2, a] * Fdagxdn₊[a, p3, p4]
    Fdagzdn₋, Fxdn₋ = Z2U1tJ2orb.FdagzdnFxdn₋
    @tensor fdagfzxdn₋[p1, p3; p2, p4] := Zzdn[p1, p1in] * Fdagzdn₋[p1in, p2, a] * Fxdn₋[a, p3, p4]
    Fzdn₋, Fdagxdn₋ = Z2U1tJ2orb.FzdnFdagxdn₋
    @tensor ffdagzxdn₋[p1, p3; p2, p4] := Zzdn[p1, p1in] * Fzdn₋[p1in, p2, a] * Fdagxdn₋[a, p3, p4]

    # intralayer Sx ⋅ Sx interaction
    SxupSxup = Z2U1tJ2orb.SxupSxup
    SxdnSxdn = Z2U1tJ2orb.SxdnSxdn

    SxupSzup = Z2U1tJ2orb.SxupSzup
    SxdnSzdn = Z2U1tJ2orb.SxdnSzdn
    FdagzupFzdn₊ = Z2U1tJ2orb.FdagzupFzdn₊
    FzupFdagzdn₊ = Z2U1tJ2orb.FzupFdagzdn₊
    FdagzupFzdn₋ = Z2U1tJ2orb.FdagzupFzdn₋
    FzupFdagzdn₋ = Z2U1tJ2orb.FzupFdagzdn₋
    SzupSzdn = Z2U1tJ2orb.SzupSzdn
    nzupnzdn = Z2U1tJ2orb.nzupnzdn
    nxup = Z2U1tJ2orb.nxup
    nxdn = Z2U1tJ2orb.nxdn
    nzup = Z2U1tJ2orb.nzup
    nzdn = Z2U1tJ2orb.nzdn
    Id = Z2U1tJ2orb.Id
    # 这里单点项要➗4, 注意区分杂化x/y方向符号相反
    gateNNx = -tc * (fdagfxup₊ - ffdagxup₊ + fdagfxdn₊ - ffdagxdn₊ + fdagfxup₋ - ffdagxup₋ + fdagfxdn₋ - ffdagxdn₋) +
              Jc * (SxupSxup - 0.25 * nxup ⊗ nxup + SxdnSxdn - 0.25 * nxdn ⊗ nxdn) +
              -td * (fdagfzup₊ - ffdagzup₊ + fdagfzdn₊ - ffdagzdn₊ + fdagfzup₋ - ffdagzup₋ + fdagfzdn₋ - ffdagzdn₋) +
              -V * (fdagfxzup₊ - ffdagxzup₊ + fdagfxzdn₊ - ffdagxzdn₊ + fdagfzxup₊ - ffdagzxup₊ + fdagfzxdn₊ - ffdagzxdn₊ + fdagfxzup₋ - ffdagxzup₋ + fdagfxzdn₋ - ffdagxzdn₋ + fdagfzxup₋ - ffdagzxup₋ + fdagfzxdn₋ - ffdagzxdn₋) +
              -0.25 * JH * ((SxupSzup + SxdnSzdn) ⊗ Id + Id ⊗ (SxupSzup + SxdnSzdn)) +
              -0.25 * tperp * ((FdagzupFzdn₊ - FzupFdagzdn₊ + FdagzupFzdn₋ - FzupFdagzdn₋) ⊗ Id + Id ⊗ (FdagzupFzdn₊ - FzupFdagzdn₊ + FdagzupFzdn₋ - FzupFdagzdn₋)) +
              0.25 * Jperp * ((SzupSzdn - 0.25 * nzupnzdn) ⊗ Id + Id ⊗ (SzupSzdn - 0.25 * nzupnzdn)) +
              0.25 * εx * ((nxup + nxdn) ⊗ Id + Id ⊗ (nxup + nxdn)) +
              0.25 * εz * ((nzup + nzdn) ⊗ Id + Id ⊗ (nzup + nzdn))

    gateNNy = -tc * (fdagfxup₊ - ffdagxup₊ + fdagfxdn₊ - ffdagxdn₊ + fdagfxup₋ - ffdagxup₋ + fdagfxdn₋ - ffdagxdn₋) +
              Jc * (SxupSxup - 0.25 * nxup ⊗ nxup + SxdnSxdn - 0.25 * nxdn ⊗ nxdn) +
              -td * (fdagfzup₊ - ffdagzup₊ + fdagfzdn₊ - ffdagzdn₊ + fdagfzup₋ - ffdagzup₋ + fdagfzdn₋ - ffdagzdn₋) +
              V * (fdagfxzup₊ - ffdagxzup₊ + fdagfxzdn₊ - ffdagxzdn₊ + fdagfzxup₊ - ffdagzxup₊ + fdagfzxdn₊ - ffdagzxdn₊ + fdagfxzup₋ - ffdagxzup₋ + fdagfxzdn₋ - ffdagxzdn₋ + fdagfzxup₋ - ffdagzxup₋ + fdagfzxdn₋ - ffdagzxdn₋) +
              -0.25 * JH * ((SxupSzup + SxdnSzdn) ⊗ Id + Id ⊗ (SxupSzup + SxdnSzdn)) +
              -0.25 * tperp * ((FdagzupFzdn₊ - FzupFdagzdn₊ + FdagzupFzdn₋ - FzupFdagzdn₋) ⊗ Id + Id ⊗ (FdagzupFzdn₊ - FzupFdagzdn₊ + FdagzupFzdn₋ - FzupFdagzdn₋)) +
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
    elseif tag == "SxupSxup_intra"
        # intralayer Sx ⋅ Sx interaction
        return Z2U1tJ2orb.SxupSxup
    elseif tag == "SxdnSxdn_intra"
        return Z2U1tJ2orb.SxdnSxdn
    elseif tag == "SzupSzup_intra"
        # intralayer Sz ⋅ Sz interaction
        return Z2U1tJ2orb.SzupSzup
    elseif tag == "SzdnSzdn_intra"
        return Z2U1tJ2orb.SzdnSzdn
    elseif tag == "SxupSzup_intra"
        # intralayer Sx ⋅ Sx interaction
        return Z2U1tJ2orb.SxupSzup
    elseif tag == "SxdnSzdn_intra"
        return Z2U1tJ2orb.SxdnSzdn
        ## single site Obs
    elseif tag == "Δₛx"
        return Z2U1tJ2orb.Δₛx
    elseif tag == "Δₛdagx"
        return Z2U1tJ2orb.Δₛdagx
    elseif tag == "Δₛxz"
        return Z2U1tJ2orb.Δₛxz
    elseif tag == "Δₛdagxz"
        return Z2U1tJ2orb.Δₛdagxz
    elseif tag == "Δₛzx"
        return Z2U1tJ2orb.Δₛzx
    elseif tag == "Δₛdagzx"
        return Z2U1tJ2orb.Δₛdagzx
    elseif tag == "Δₛz"
        return Z2U1tJ2orb.Δₛz
    elseif tag == "Δₛdagz"
        return Z2U1tJ2orb.Δₛdagz
    elseif tag == "Nxup"
        return Z2U1tJ2orb.nxup
    elseif tag == "Nxdn"
        return Z2U1tJ2orb.nxdn
    elseif tag == "Nzup"
        return Z2U1tJ2orb.nzup
    elseif tag == "Nzdn"
        return Z2U1tJ2orb.nzdn
    elseif tag == "Sᶻxup"
        return Z2U1tJ2orb.Sᶻxup
    elseif tag == "Sᶻxdn"
        return Z2U1tJ2orb.Sᶻxdn
    elseif tag == "Sᶻzup"
        return Z2U1tJ2orb.Sᶻzup
    elseif tag == "Sᶻzdn"
        return Z2U1tJ2orb.Sᶻzdn
    elseif tag == "SxupSzup"
        return Z2U1tJ2orb.SxupSzup
    elseif tag == "SxdnSzdn"
        return Z2U1tJ2orb.SxdnSzdn
    elseif tag == "SzupSzdn"
        return Z2U1tJ2orb.SzupSzdn
    elseif tag == "SxupSxdn"
        return Z2U1tJ2orb.SxupSxdn
    elseif tag == "nzupnzdn"
        return Z2U1tJ2orb.nzupnzdn
    else
        error("Unsupported tag $tag. Check input tag or add this operator.")
    end
end