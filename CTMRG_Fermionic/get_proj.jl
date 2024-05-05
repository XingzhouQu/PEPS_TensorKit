function get_proj_update_left(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, x::Int, y::Int, χ::Int)
    # Qu 上面一半  Qd 下面一半
    QuL = get_QuL(ipeps, ipepsbar, envs, x, y)
    QuR = get_QuR(ipeps, ipepsbar, envs, x + 1, y)
    QdL = get_QdL(ipeps, ipepsbar, envs, x, y + 1)
    QdR = get_QdR(ipeps, ipepsbar, envs, x + 1, y + 1)
    # 复杂度最高的两步缩并
    @tensor Qu[(); (bχL, bupDL, bdnDL, bχR, bupDR, bdnDR)] :=
        QuL[rχin, rupD, rdnD, bχL, bupDL, bdnDL] * QuR[rχin, rupD, rdnD, bχR, bupDR, bdnDR]
    @tensor Qd[(tχL, tupDL, tdnDL, tχR, tupDR, tdnDR); ()] :=
        QdL[tχL, tupDL, tdnDL, rχin, rupD, rdnD] * QdR[rχin, rupD, rdnD, tχR, tupDR, tdnDR]

    # LQ 分解. Q 是正交基  @assert Vudag * Vudag' ≈ id(codomain(Vudag))
    Ru, Vudag = rightorth(Qu, ((1, 2, 3), (4, 5, 6)))
    Rd, Vddag = rightorth(Qd, ((1, 2, 3), (4, 5, 6)))
    @tensor Rud[t; b] := Ru[χL, upDL, dnDL, t] * Rd[χL, upDL, dnDL, b]
    U, S, Vdag, ϵ = tsvd(Rud, ((1,), (2,)); trunc=truncdim(χ), alg=TensorKit.SVD())  # SVD() is more stable
    S_invHalf = SqrtInv(S)
    # printMinMax(S_invHalf)
    # RevdnupD = isomorphism(dual(space(Ru)[2]), space(Ru)[2])
    # RevdndnD = isomorphism(dual(space(Ru)[3]), space(Ru)[3])
    @tensor projup[χ, upD, dnD; toU] := (S_invHalf[toV, toU] * Vdag'[toRd, toV]) * Rd[χ, upD, dnD, toRd]
    @tensor projdn[(toV); (χ, upD, dnD)] := S_invHalf[toV, toU] * U'[toU, toRu] * Ru[χ, upD, dnD, toRu] #* RevdnupD[upDflip, upD] * RevdndnD[dnDflip, dnD]

    return projup / norm(projup, Inf), projdn / norm(projdn, Inf), ϵ
end

function get_proj_update_right(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, x::Int, y::Int, χ::Int)
    # Qu 上面一半  Qd 下面一半
    QuL = get_QuL(ipeps, ipepsbar, envs, x, y)
    QuR = get_QuR(ipeps, ipepsbar, envs, x + 1, y)
    QdL = get_QdL(ipeps, ipepsbar, envs, x, y + 1)
    QdR = get_QdR(ipeps, ipepsbar, envs, x + 1, y + 1)
    # 复杂度最高的两步缩并
    @tensor Qu[(); (bχL, bupDL, bdnDL, bχR, bupDR, bdnDR)] :=
        QuL[rχin, rupD, rdnD, bχL, bupDL, bdnDL] * QuR[rχin, rupD, rdnD, bχR, bupDR, bdnDR]
    @tensor Qd[(tχL, tupDL, tdnDL, tχR, tupDR, tdnDR); ()] :=
        QdL[tχL, tupDL, tdnDL, rχin, rupD, rdnD] * QdR[rχin, rupD, rdnD, tχR, tupDR, tdnDR]

    # QR 分解. Q 是正交基  @assert Uu' * Uu ≈ id(domain(Uu))
    Uu, Ru = leftorth(Qu, ((1, 2, 3), (4, 5, 6)))
    Ud, Rd = leftorth(Qd, ((1, 2, 3), (4, 5, 6)))
    @tensor Rud[t; b] := Ru[t, χL, upDL, dnDL] * Rd[b, χL, upDL, dnDL]
    U, S, Vdag, ϵ = tsvd(Rud, ((1,), (2,)); trunc=truncdim(χ), alg=TensorKit.SVD())
    S_invHalf = SqrtInv(S)
    # printMinMax(S_invHalf)
    # RevdnupD = isomorphism(dual(space(Ru)[3]), space(Ru)[3])
    # RevdndnD = isomorphism(dual(space(Ru)[4]), space(Ru)[4])
    @tensor projup[χ, upD, dnD; toU] := (S_invHalf[toV, toU] * Vdag'[toRd, toV]) * Rd[toRd, χ, upD, dnD]  # ∇
    @tensor projdn[(toV); (χL, upD, dnD)] := # Δ
        S_invHalf[toV, toU] * U'[toU, toRu] * Ru[toRu, χL, upD, dnD] #* RevdnupD[upDflip, upD] * RevdndnD[dnDflip, dnD]

    return projup / norm(projup, Inf), projdn / norm(projdn, Inf), ϵ
end


function get_proj_update_top(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, x::Int, y::Int, χ::Int)
    QuL = get_QuL(ipeps, ipepsbar, envs, x, y)
    QuR = get_QuR(ipeps, ipepsbar, envs, x + 1, y)
    QdL = get_QdL(ipeps, ipepsbar, envs, x, y + 1)
    QdR = get_QdR(ipeps, ipepsbar, envs, x + 1, y + 1)
    # 复杂度最高的缩并
    @tensor QL[(); (rχt, rupDt, rdnDt, rχb, rupDb, rdnDb)] :=
        QuL[rχt, rupDt, rdnDt, bχLin, bupDL, bdnDL] * QdL[bχLin, bupDL, bdnDL, rχb, rupDb, rdnDb]
    @tensor QR[(lχt, lupDt, ldnDt, lχb, lupDb, ldnDb); ()] :=
        QuR[lχt, lupDt, ldnDt, bχin, bupD, bdnD] * QdR[lχb, lupDb, ldnDb, bχin, bupD, bdnD]

    # LQ 分解. Q 是正交基  @assert Vldag * Vldag' ≈ id(codomain(Vldag))
    Rl, Vldag = rightorth(QL, ((1, 2, 3), (4, 5, 6)))
    Rr, Vrdag = rightorth(QR, ((1, 2, 3), (4, 5, 6)))
    @tensor Rlr[l; r] := Rl[χ, upD, dnD, l] * Rr[χ, upD, dnD, r]
    U, S, Vdag, ϵ = tsvd(Rlr, ((1,), (2,)); trunc=truncdim(χ), alg=TensorKit.SVD())
    S_invHalf = SqrtInv(S)
    # printMinMax(S_invHalf)

    @tensor projleft[(toU); (χ, upD, dnD)] := S_invHalf[toV, toU] * Vdag'[toRr, toV] * Rr[χ, upD, dnD, toRr]  # ▷
    @tensor projright[(χ, upD, dnD); (toV)] := S_invHalf[toV, toU] * U'[toU, toRl] * Rl[χ, upD, dnD, toRl]  # ◁

    return projleft / norm(projleft, Inf), projright / norm(projright, Inf), ϵ
end

function get_proj_update_bottom(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, x::Int, y::Int, χ::Int)
    QuL = get_QuL(ipeps, ipepsbar, envs, x, y)
    QuR = get_QuR(ipeps, ipepsbar, envs, x + 1, y)
    QdL = get_QdL(ipeps, ipepsbar, envs, x, y + 1)
    QdR = get_QdR(ipeps, ipepsbar, envs, x + 1, y + 1)
    # 复杂度最高的缩并
    @tensor QL[(rχt, rχb); (rupDt, rdnDt, rupDb, rdnDb)] :=
        QuL[rχt, rupDt, rdnDt, bχLin, bupDL, bdnDL] * QdL[bχLin, bupDL, bdnDL, rχb, rupDb, rdnDb]
    @tensor QR[(lupDt, ldnDt, lupDb, ldnDb); (lχt, lχb)] :=
        QuR[lχt, lupDt, ldnDt, bχin, bupD, bdnD] * QdR[lχb, lupDb, ldnDb, bχin, bupD, bdnD]

    # QR 分解. Q 是正交基  @assert Ul' * Ul ≈ id(domain(Uu))
    Ul, Rl = leftorth(QL, ((1, 3, 4), (2, 5, 6)))
    Ur, Rr = leftorth(QR, ((5, 1, 2), (6, 3, 4)))
    @tensor Rlr[l; r] := Rl[l, χ, upDL, dnDL] * Rr[r, χ, upDL, dnDL]
    U, S, Vdag, ϵ = tsvd(Rlr, ((1,), (2,)); trunc=truncdim(χ), alg=TensorKit.SVD())
    S_invHalf = SqrtInv(S)
    # printMinMax(S_invHalf)

    @tensor projleft[(toU); (χ, upD, dnD)] := S_invHalf[toV, toU] * Vdag'[toRr, toV] * Rr[toRr, χ, upD, dnD]  # ▷
    @tensor projright[(χ, upD, dnD); (toV)] := S_invHalf[toV, toU] * U'[toU, toRl] * Rl[toRl, χ, upD, dnD]  # ◁

    return projleft / norm(projleft, Inf), projright / norm(projright, Inf), ϵ
end


function get_QuL(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, x::Int, y::Int)
    # 左上角半边
    C = envs[x, y].corner.lt
    Tt = envs[x, y].transfer.t
    Tl = envs[x, y].transfer.l
    M = ipeps[x, y]
    Mbar = ipepsbar[x, y]
    gate1 = swap_gate(space(Tt)[4], space(Tl)[3]; Eltype=eltype(M))
    gate2 = swap_gate(space(M)[5], space(Mbar)[4]; Eltype=eltype(M))

    @tensor CTtTlMMbar[(); (rχ, rupMD, rdnMD, bχ, bupMD, bdnMD)] :=
        Tt[rχin, rχ, bupDin, rupDMin1] * C[rχin, bχin] * Tl[bχin, bχ, rupDMin2, rdnD] *
        gate1[rupDin1, rupDin2, rupDMin1, rupDMin2] * M[rupDin2, bupDin, p, rupMD, bupMDin] *
        Mbar[rdnD, rupDin1, p, rdnMDin, bdnMD] * gate2[bupMD, rdnMD, bupMDin, rdnMDin]
    return CTtTlMMbar
end

function get_QuR(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, x::Int, y::Int)
    # 右上角半边
    C = envs[x, y].corner.rt
    Tt = envs[x, y].transfer.t
    Tr = envs[x, y].transfer.r
    M = ipeps[x, y]
    Mbar = ipepsbar[x, y]
    gate1 = swap_gate(space(M)[1], space(Mbar)[2]; Eltype=eltype(M))
    gate2 = swap_gate(space(M)[5], space(Mbar)[4]; Eltype=eltype(M))

    @tensor CTtTrMMbar[lχ, lupD, ldnD; bχ, bupD, bdnD] :=
        C[lχin, bχin] * Tt[lχ, lχin, bupDin, bdnDin] * Tr[lupDin, ldnDin, bχin, bχ] *
        M[lupDMin, bupDin, p, lupDin, bupDMin] * gate1[lupD, bdnDin, lupDMin, bdnDMin] *
        gate2[bupD, ldnDin, bupDMin, ldnDMin] * Mbar[ldnD, bdnDMin, p, ldnDMin, bdnD]
    return CTtTrMMbar
end

function get_QdL(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, x::Int, y::Int)
    # 左下角半边
    C = envs[x, y].corner.lb
    Tl = envs[x, y].transfer.l
    Tb = envs[x, y].transfer.b
    M = ipeps[x, y]
    Mbar = ipepsbar[x, y]
    gate1 = swap_gate(space(M)[1], space(Mbar)[2]; Eltype=eltype(M))
    gate2 = swap_gate(space(M)[5], space(Mbar)[4]; Eltype=eltype(M))

    @tensor CTlTbMbarM[tχ, tupD, tdnD; rχ, rupD, rdnD] :=
        C[tχin, rχin] * Tl[tχ, tχin, rupDin, rdnDin] * Tb[rχin, tupDin, tdnDin, rχ] *
        Mbar[rdnDin, tdnDMin, p, rdnDMin, tdnDin] * gate2[tupDin, rdnD, tupDMin, rdnDMin] *
        gate1[rupDin, tdnD, rupDMin, tdnDMin] * M[rupDMin, tupD, p, rupD, tupDMin]
    return CTlTbMbarM
end

function get_QdR(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, x::Int, y::Int)
    # 右下角半边
    C = envs[x, y].corner.rb
    Tr = envs[x, y].transfer.r
    Tb = envs[x, y].transfer.b
    M = ipeps[x, y]
    Mbar = ipepsbar[x, y]
    gate1 = swap_gate(space(M)[1], space(Mbar)[2]; Eltype=eltype(M))
    gate2 = swap_gate(space(M)[5], space(Mbar)[4]; Eltype=eltype(M))

    @tensor CTrTbMbarM[(lχ, lupD, ldnD, tχ, tupD, tdnD); ()] :=
        C[lχin, tχin] * Tr[lupMDin, ldnDin, tχ, tχin] * Tb[lχ, tupDin, tdnDin, lχin] *
        gate2[tupDin, ldnDin, tupDMin, ldnDMin] * Mbar[ldnD, tdnDMin, p, ldnDMin, tdnDin] *
        M[lupDMin, tupD, p, lupMDin, tupDMin] * gate1[lupD, tdnD, lupDMin, tdnDMin]
    return CTrTbMbarM
end