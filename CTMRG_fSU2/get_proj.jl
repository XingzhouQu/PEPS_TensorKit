function get_proj_update_LR(ipeps::iPEPS, envs::iPEPSenv, x::Int, y::Int, χ::Int; kwargs...)
    direction = get(kwargs, :dir, "left")

    # Qu 上面一半  Qd 下面一半
    QuL = get_QuL(ipeps, envs, x, y)
    QuR = get_QuR(ipeps, envs, x + 1, y)
    QdL = get_QdL(ipeps, envs, x, y + 1)
    QdR = get_QdR(ipeps, envs, x + 1, y + 1)
    # 复杂度最高的两步缩并
    @tensor Qu[(); (bχL, bupDL, bdnDL, bχR, bupDR, bdnDR)] :=
        QuL[rχin, rupD, rdnD, bχL, bupDL, bdnDL] * QuR[rχin, rupD, rdnD, bχR, bupDR, bdnDR]
    @tensor Qd[(tχL, tupDL, tdnDL, tχR, tupDR, tdnDR); ()] :=
        QdL[tχL, tupDL, tdnDL, rχin, rupD, rdnD] * QdR[rχin, rupD, rdnD, tχR, tupDR, tdnDR]

    if direction == "left"
        # LQ 分解. Q 是正交基  @assert Vudag * Vudag' ≈ id(codomain(Vudag))
        Ru, Vudag = rightorth(Qu, ((1, 2, 3), (4, 5, 6)))
        Rd, Vddag = rightorth(Qd, ((1, 2, 3), (4, 5, 6)))
        @tensor Rud[t; b] := Ru[χL, upDL, dnDL, t] * Rd[χL, upDL, dnDL, b]
        U, S, Vdag, ϵ = tsvd(Rud, ((1,), (2,)); trunc=truncdim(χ))
        S_inv_sqrt = inv_sqrt(S)
        # RevdnupD = isomorphism(dual(space(Ru)[2]), space(Ru)[2])
        # RevdndnD = isomorphism(dual(space(Ru)[3]), space(Ru)[3])
        @tensor projup[χ, upD, dnD; toU] := (S_inv_sqrt[toV, toU] * Vdag'[toRd, toV]) * Rd[χ, upD, dnD, toRd]
        @tensor projdn[(toV); (χ, upD, dnD)] :=
            (S_inv_sqrt[toV, toU] * U'[toU, toRu]) * Ru[χ, upD, dnD, toRu] #* RevdnupD[upDflip, upD] * RevdndnD[dnDflip, dnD]
    elseif direction == "right"
        # QR 分解. Q 是正交基  @assert Uu' * Uu ≈ id(domain(Uu))
        Uu, Ru = leftorth(Qu, ((1, 2, 3), (4, 5, 6)))
        Ud, Rd = leftorth(Qd, ((1, 2, 3), (4, 5, 6)))
        @tensor Rud[t; b] := Ru[t, χL, upDL, dnDL] * Rd[b, χL, upDL, dnDL]
        U, S, Vdag, ϵ = tsvd(Rud, ((1,), (2,)); trunc=truncdim(χ))
        S_inv_sqrt = inv_sqrt(S)
        # RevdnupD = isomorphism(dual(space(Ru)[3]), space(Ru)[3])
        # RevdndnD = isomorphism(dual(space(Ru)[4]), space(Ru)[4])
        @tensor projup[χ, upD, dnD; toU] := (S_inv_sqrt[toV, toU] * Vdag'[toRd, toV]) * Rd[toRd, χ, upD, dnD]  # ∇
        @tensor projdn[(toV); (χL, upD, dnD)] := # Δ
            (S_inv_sqrt[toV, toU] * U'[toU, toRu]) * Ru[toRu, χL, upD, dnD] #* RevdnupD[upDflip, upD] * RevdndnD[dnDflip, dnD]
    else
        error("kwargs for get_proj_update_LR() should be `left` or `right`")
    end
    return projup, projdn, ϵ
end


function get_proj_update_UD(ipeps::iPEPS, envs::iPEPSenv, x::Int, y::Int, χ::Int; kwargs...)
    direction = get(kwargs, :dir, "up")

    QuL = get_QuL(ipeps, envs, x, y)
    QuR = get_QuR(ipeps, envs, x + 1, y)
    QdL = get_QdL(ipeps, envs, x, y + 1)
    QdR = get_QdR(ipeps, envs, x + 1, y + 1)
    # 复杂度最高的缩并
    @tensor QL[(rχt, rχb); (rupDt, rdnDt, rupDb, rdnDb)] :=
        QuL[rχt, rupDt, rdnDt, bχLin, bupDL, bdnDL] * QdL[bχLin, bupDL, bdnDL, rχb, rupDb, rdnDb]
    @tensor QR[(lupDt, ldnDt, lupDb, ldnDb); (lχt, lχb)] :=
        QuR[lχt, lupDt, ldnDt, bχin, bupD, bdnD] * QdR[lχb, lupDb, ldnDb, bχin, bupD, bdnD]

    if direction == "up"
        # LQ 分解. Q 是正交基  @assert Vldag * Vldag' ≈ id(codomain(Vldag))
        Rl, Vldag = rightorth(QL, ((1, 3, 4), (2, 5, 6)))
        Rr, Vrdag = rightorth(QR, ((5, 1, 2), (6, 3, 4)))
        @tensor Rlr[l; r] := Rl[χ, upD, dnD, l] * Rr[χ, upD, dnD, r]
        U, S, Vdag, ϵ = tsvd(Rlr, ((1,), (2,)); trunc=truncdim(χ))
        S_inv_sqrt = inv_sqrt(S)
        # 这里需要手动把两个QR分解出来的指标变成对偶空间
        Revleftχ = isomorphism(dual(space(Rl)[1]), space(Rl)[1])
        RevleftupD = isomorphism(dual(space(Rl)[2]), space(Rl)[2])
        RevleftdnD = isomorphism(dual(space(Rl)[3]), space(Rl)[3])
        Revrightχ = isomorphism(dual(space(Rr)[1]), space(Rr)[1])
        RevrightupD = isomorphism(dual(space(Rr)[2]), space(Rr)[2])
        RevrightdnD = isomorphism(dual(space(Rr)[3]), space(Rr)[3])
        @tensor projleft[(toU); (χ, upD, dnD)] :=  # ▷
            (S_inv_sqrt[toV, toU] * Vdag'[toRr, toV]) * Rl[χflip, upDflip, dnDflip, toRr] * Revleftχ[χflip, χ] *
            RevleftupD[upDflip, upD] * RevleftdnD[dnDflip, dnD]
        @tensor projright[(χ, upD, dnD); (toV)] :=  # ◁
            (S_inv_sqrt[toV, toU] * U'[toU, toRl]) * Rr[χflip, upDflip, dnDflip, toRl] * Revrightχ[χflip, χ] *
            RevrightupD[upDflip, upD] * RevrightdnD[dnDflip, dnD]
    elseif direction == "dn"
        # QR 分解. Q 是正交基  @assert Ul' * Ul ≈ id(domain(Uu))
        Ul, Rl = leftorth(QL, ((1, 3, 4), (2, 5, 6)))
        Ur, Rr = leftorth(QR, ((5, 1, 2), (6, 3, 4)))
        @tensor Rlr[l; r] := Rl[l, χ, upDL, dnDL] * Rr[r, χ, upDL, dnDL]
        U, S, Vdag, ϵ = tsvd(Rlr, ((1,), (2,)); trunc=truncdim(χ))
        S_inv_sqrt = inv_sqrt(S)

        Revleftχ = isomorphism(dual(space(Rl)[2]), space(Rl)[2])
        RevleftupD = isomorphism(dual(space(Rl)[3]), space(Rl)[3])
        RevleftdnD = isomorphism(dual(space(Rl)[4]), space(Rl)[4])
        Revrightχ = isomorphism(dual(space(Rr)[2]), space(Rr)[2])
        RevrightupD = isomorphism(dual(space(Rr)[3]), space(Rr)[3])
        RevrightdnD = isomorphism(dual(space(Rr)[4]), space(Rr)[4])
        @tensor projleft[(toU); (χ, upD, dnD)] := # ▷
            (S_inv_sqrt[toV, toU] * Vdag'[toRr, toV]) * Rl[toRr, χflip, upDflip, dnDflip] * Revleftχ[χflip, χ] *
            RevleftupD[upDflip, upD] * RevleftdnD[dnDflip, dnD]
        @tensor projright[(χ, upD, dnD); (toV)] := # ◁
            (S_inv_sqrt[toV, toU] * U'[toU, toRl]) * Rr[toRl, χflip, upDflip, dnDflip] * Revrightχ[χflip, χ] *
            RevrightupD[upDflip, upD] * RevrightdnD[dnDflip, dnD]
    else
        error("kwargs for get_proj_update_UD() should be `up` or `dn`")
    end
    return projleft, projright, ϵ
end


function get_QuL(ipeps::iPEPS, envs::iPEPSenv, x::Int, y::Int)
    # 左上角半边
    C = envs[x, y].corner.lt
    Tr = envs[x, y].transfer.t
    Tb = envs[x, y].transfer.l
    M = ipeps[x, y]
    Mbar = M'

    @tensor CTr[bχ, rχ, bupD, bdnD] := Tr[rχin, rχ, bupD, bdnD] * C[rχin, bχ]
    @tensor CTrTb[rχ, bupD, bdnD, rupD, rdnD, bχ] := CTr[bχin, rχ, bupD, bdnD] * Tb[bχin, bχ, rupD, rdnD]
    @tensor CTrTbM[rχ, bχ, rupMD, bupMD, rupD, rdnD, p] :=
        CTrTb[rχ, bupDin, rupD, rupDin, rdnD, bχ] * M[rupDin, bupDin, p, rupMD, bupMD]
    @tensor CTrTbMMbar[(); (rχ, rupMD, rdnMD, bχ, bupMD, bdnMD)] :=
        CTrTbM[rχ, bχ, rupMD, bupMD, rupD, rdnD, p] * Mbar[rdnMD, bdnMD, rdnD, rupD, p]
    return CTrTbMMbar
end

function get_QuR(ipeps::iPEPS, envs::iPEPSenv, x::Int, y::Int)
    # 右上角半边
    C = envs[x, y].corner.rt
    Tl = envs[x, y].transfer.t
    Tb = envs[x, y].transfer.r
    M = ipeps[x, y]
    Mbar = M'

    @tensor CTl[lχ, bχ, bupD, bdnD] := C[lχin, bχ] * Tl[lχ, lχin, bupD, bdnD]
    @tensor CTlTb[lχ, lupD, ldnD, bχ, bupD, bdnD] := CTl[lχ, bχin, bupD, bdnD] * Tb[lupD, ldnD, bχin, bχ]
    @tensor CTlTbM[lχ, lupD, ldnD, bχ, bupD, bdnD, p] :=
        CTlTb[lχ, lupDin, ldnD, bχ, bupDin, bdnD] * M[lupD, bupDin, p, lupDin, bupD]
    @tensor CTlTbMMbar[lχ, lupD, ldnD; bχ, bupD, bdnD] :=
        CTlTbM[lχ, lupD, ldnDin, bχ, bupD, bdnDin, p] * Mbar[ldnDin, bdnD, ldnD, bdnDin, p]
    return CTlTbMMbar
end

function get_QdL(ipeps::iPEPS, envs::iPEPSenv, x::Int, y::Int)
    # 左下角半边
    C = envs[x, y].corner.lb
    Tl = envs[x, y].transfer.l
    Tb = envs[x, y].transfer.b
    M = ipeps[x, y]
    Mbar = M'

    @tensor CTl[tχ, rχ, rupD, rdnD] := C[tχin, rχ] * Tl[tχ, tχin, rupD, rdnD]
    @tensor CTlTb[tχ, rupD, rdnD, rχ, tupD, tdnD] := CTl[tχ, rχin, rupD, rdnD] * Tb[rχin, tupD, tdnD, rχ]
    @tensor CTlTbMbar[tχ, rupD, rdnD, rχ, tupD, tdnD, p] :=
        CTlTb[tχ, rupD, rdnDin, rχ, tupD, tdnDin] * Mbar[rdnD, tdnDin, rdnDin, tdnD, p]
    @tensor CTlTbMbarM[tχ, tupD, tdnD; rχ, rupD, rdnD] :=
        CTlTbMbar[tχ, rupDin, rdnD, rχ, tupDin, tdnD, p] * M[rupDin, tupD, p, rupD, tupDin]
    return CTlTbMbarM
end

function get_QdR(ipeps::iPEPS, envs::iPEPSenv, x::Int, y::Int)
    # 右下角半边
    C = envs[x, y].corner.rb
    Tr = envs[x, y].transfer.r
    Tb = envs[x, y].transfer.b
    M = ipeps[x, y]
    Mbar = M'

    @tensor CTr[lχ, tχ, lupD, ldnD] := C[lχ, tχin] * Tr[lupD, ldnD, tχ, tχin]
    @tensor CTrTb[tχ, lupD, ldnD, lχ, tupD, tdnD] := CTr[lχin, tχ, lupD, ldnD] * Tb[lχ, tupD, tdnD, lχin]
    @tensor CTrTbMbar[tχ, lupD, ldnD, lχ, tupD, tdnD, p] :=
        CTrTb[tχ, lupD, ldnDin, lχ, tupD, tdnDin] * Mbar[ldnDin, tdnDin, ldnD, tdnD, p]
    @tensor CTrTbMbarM[(lχ, lupD, ldnD, tχ, tupD, tdnD); ()] :=
        CTrTbMbar[tχ, lupMDin, ldnD, lχ, tupDin, tdnD, p] * M[lupD, tupD, p, lupMDin, tupDin]
    return CTrTbMbarM
end