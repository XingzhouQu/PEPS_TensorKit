# using ChainRulesCore: ignore_derivatives
include("../iPEPS_Fermionic/swap_gate.jl")

function Cal_Obs_1site(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, Ops::Vector{String}, para::Dict{Symbol,Any}; site=[1, 1], get_op::Function)
    x = site[1]
    y = site[2]
    vals = Vector{Number}(undef, length(Ops))
    M = ipeps[x, y]
    Mbar = ipepsbar[x, y]
    gate1 = swap_gate(space(M)[1], space(Mbar)[2]; Eltype=eltype(M))
    gate2 = swap_gate(space(M)[5], space(Mbar)[4]; Eltype=eltype(M))
    println("Fermionic! Calculating $Ops at site $site")
    # 顺序：CTTM?̄MCCTCT
    @tensor opt = true ψ□ψ[pup, pdn] :=
        envs[x, y].corner.lt[toT, toL] * envs[x, y].transfer.t[toT, toRT, toMtup, toMtdn] *
        envs[x, y].transfer.l[toL, toLB, toMlup, toMldn] * gate1[toMlup, toMtdn, toMlupin, toMtdnin] *
        M[toMlupin, toMtup, pup, toMrup, toMbupin] * Mbar[toMldn, toMtdnin, pdn, toMrdnin, toMbdn] *
        gate2[toMbup, toMrdn, toMbupin, toMrdnin] * envs[x, y].corner.rt[toRT, toR] *
        envs[x, y].corner.lb[toLB, toB] * envs[x, y].transfer.r[toMrup, toMrdn, toR, toRB] *
        envs[x, y].transfer.b[toB, toMbup, toMbdn, btoRB] * envs[x, y].corner.rb[btoRB, toRB]
    @tensor nrm = ψ□ψ[p, p]
    for (ii, tag) in enumerate(Ops)
        op = get_op(tag, para)
        @tensor tmp = ψ□ψ[pup, pdn] * op[pdn, pup]
        vals[ii] = tmp
    end
    rslt = Dict(Ops[ind] => (vals[ind] / nrm) for ind in 1:length(vals))
    return rslt
end

"""
计算两点观测量。输入的site convention:\n
[site1, site2] = \n
[左,右]; [上,下]; [左上,右下]; [右上,左下]
"""
function Cal_Obs_2site(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, Gates::Vector{String}, para::Dict{Symbol,Any};
    site1::Vector{Int}, site2::Vector{Int}, get_op::Function)
    Lx = ipeps.Lx
    Ly = ipeps.Ly
    @assert Lx >= 2 || Ly >= 2 "The unit cell should contain no less than two sites"
    if 0.99 < norm(site1 - site2) < 1.01  # 近邻格点
        println("Fermionic! Calculating $Gates at site $site1 and site $site2")
        rslt = _2siteObs_adjSite(ipeps, ipepsbar, envs, Gates, para, site1, site2, get_op)
        return rslt
    elseif (abs(site1[1] - site2[1]) == 1) && (abs(site1[2] - site2[2]) == 1)  # 斜对角格点
        println("Fermionic! Calculating $Gates at site $site1 and site $site2")
        rslt = _2siteObs_diagSite(ipeps, ipepsbar, envs, Gates, para, site1, site2, get_op)
        return rslt
    else
        error("Larger distance is not supported yet.")
    end
    return nothing
end


function _2siteObs_adjSite(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, Gates::Vector{String}, para::Dict{Symbol,Any},
    site1::Vector{Int}, site2::Vector{Int}, get_op::Function; ADflag=false)
    # ADflag ? ipepsbar = ignore_derivatives(ipepsbar) : nothing
    x1, y1 = site1
    x2, y2 = site2
    M1 = ipeps[x1, y1]
    M1bar = ipepsbar[x1, y1]
    M2 = ipeps[x2, y2]
    M2bar = ipepsbar[x2, y2]
    if y2 == y1  # 横向的两个点.
        swgatel1 = swap_gate(space(M1)[1], space(M1bar)[2]; Eltype=eltype(M1))
        swgatel2 = swap_gate(space(M1)[3], space(M1)[5]; Eltype=eltype(M1))
        swgatel3 = swap_gate(space(M1bar)[3], space(swgatel2)[2]; Eltype=eltype(M1))
        swgatel4 = swap_gate(space(swgatel3)[2], space(M1bar)[4]; Eltype=eltype(M1))
        swgater4 = swap_gate(space(M2)[5], space(M2bar)[4]; Eltype=eltype(M1))
        swgater3 = swap_gate(space(M2bar)[3], space(M2bar)[2]; Eltype=eltype(M1))
        swgater2 = swap_gate(space(M2)[3], space(swgater3)[2]; Eltype=eltype(M1))
        swgater1 = swap_gate(space(M2)[1], space(swgater2)[2]; Eltype=eltype(M1))
        # order =  (lt2t1, t12Ddn, l2Dupin, t12Dup, pup1in, b12Dupin, b12Dupin2, pdn1in, t12Ddnin, lt2l, l2Dup, l2Ddn, b12Dupin3, Ddnin2, b12Dup, b12Ddn, lb2l, lb2b1)
        @tensor opt = true leftpart[pup1, Dupin, t12t2; pdn1, Ddnin, b12b2] :=
            envs[x1, y1].corner.lt[lt2t1, lt2l] * envs[x1, y1].transfer.t[lt2t1, t12t2, t12Dup, t12Ddn] *
            swgatel1[l2Dup, t12Ddn, l2Dupin, t12Ddnin] * M1[l2Dupin, t12Dup, pup1in, Dupin, b12Dupin] *
            swgatel2[pup1, b12Dupin2, pup1in, b12Dupin] * swgatel3[pdn1, b12Dupin3, pdn1in, b12Dupin2] *
            M1bar[l2Ddn, t12Ddnin, pdn1in, Ddnin2, b12Ddn] * envs[x1, y1].transfer.l[lt2l, lb2l, l2Dup, l2Ddn] *
            swgatel4[b12Dup, Ddnin, b12Dupin3, Ddnin2] * envs[x1, y1].transfer.b[lb2b1, b12Dup, b12Ddn, b12b2] *
            envs[x1, y1].corner.lb[lb2l, lb2b1]
        @tensor opt = true rightpart[pup2, t12t2, Dupin; pdn2, Ddnin, b12b2] :=
            swgater1[Dupin, t22Ddn, Dupinin, t22Ddnin3] *
            envs[x2, y2].transfer.t[t12t2, rt2t2, t22Dup, t22Ddn] * M2[Dupinin, t22Dup, pup2in, r2Dup, b22Dupin] *
            swgater2[pup2, t22Ddnin3, pup2in, t22Ddnin2] * swgater3[pdn2, t22Ddnin2, pdn2in, t22Ddnin] *
            M2bar[Ddnin, t22Ddnin, pdn2in, r2Ddnin, b22Ddn] * envs[x2, y2].transfer.b[b12b2, b22Dup, b22Ddn, rb2b2] *
            envs[x2, y2].corner.rt[rt2t2, rt2r] * swgater4[b22Dup, r2Ddn, b22Dupin, r2Ddnin] *
            envs[x2, y2].transfer.r[r2Dup, r2Ddn, rt2r, rb2r] * envs[x2, y2].corner.rb[rb2b2, rb2r]
        @tensor opt = true ψ□ψ[pup1, pup2; pdn1, pdn2] := leftpart[pup1, Dupin, t12t2, pdn1, Ddnin, b12b2] * rightpart[pup2, t12t2, Dupin, pdn2, Ddnin, b12b2]
        # ignore_derivatives() do
        #     leftpart, rightpart = nothing, nothing
        #     for ii in 1:4
        #         eval(Meta.parse("swgatel$ii, swgater$ii = nothing, nothing"))
        #     end
        #     GC.gc()
        # end
        @tensor nrm = ψ□ψ[p1, p2, p1, p2]
    elseif x2 == x1  # 纵向的两个点
        swgatet1 = swap_gate(space(M1)[1], space(M1bar)[2]; Eltype=eltype(M1))
        swgatet2 = swap_gate(space(M1bar)[3], space(M1bar)[4]; Eltype=eltype(M1))
        swgatet3 = swap_gate(space(M1)[3], space(swgatet2)[2]; Eltype=eltype(M1))
        swgatet4 = swap_gate(space(M1)[5], space(swgatet3)[2]; Eltype=eltype(M1))
        swgateb4 = swap_gate(space(M2bar)[4], space(M2)[5]; Eltype=eltype(M1))
        swgateb3 = swap_gate(space(M2)[1], space(M2)[3]; Eltype=eltype(M1))
        swgateb2 = swap_gate(space(M2bar)[3], space(swgateb3)[1]; Eltype=eltype(M1))
        swgateb1 = swap_gate(space(M2bar)[2], space(swgateb2)[2]; Eltype=eltype(M1))
        @tensor opt = true toppart[pup1, Dupin, l12l2; pdn1, Ddnin, r12r2] :=
            envs[x1, y1].corner.lt[lt2t, lt2l1] * envs[x1, y1].transfer.l[lt2l1, l12l2, l12Dup, l12Ddn] *
            envs[x1, y1].transfer.t[lt2t, rt2t, t2Dup, t2Ddn] * envs[x1, y1].corner.rt[rt2t, rt2r1] *
            swgatet1[l12Dup, t2Ddn, l12Dupin, t2Ddnin] * M1[l12Dupin, t2Dup, pup1in, r12Dup, Dupin2] *
            M1bar[l12Ddn, t2Ddnin, pdn1in, r12Ddnin, Ddnin] * swgatet2[pdn1, r12Ddnin2, pdn1in, r12Ddnin] *
            swgatet3[pup1, r12Ddnin3, pup1in, r12Ddnin2] * swgatet4[Dupin, r12Ddn, Dupin2, r12Ddnin3] *
            envs[x1, y1].transfer.r[r12Dup, r12Ddn, rt2r1, r12r2]
        @tensor opt = true bottompart[pup2, Dupin, l12l2; pdn2, Ddnin, r12r2] :=
            envs[x2, y2].transfer.l[l12l2, lb2l2, l22Dup, l22Ddn] *
            envs[x2, y2].corner.lb[lb2l2, lb2b] * swgateb1[Ddnin, l22Dup, Ddninin, l22Dupin3] *
            M2bar[l22Ddn, Ddninin, pdn2in, r22Ddnin, b2Ddn] * swgateb2[pdn2, l22Dupin3, pdn2in, l22Dupin2] *
            swgateb3[l22Dupin2, pup2, l22Dupin, pup2in] * M2[l22Dupin, Dupin, pup2in, r22Dup, b2Dupin] *
            swgateb4[r22Ddn, b2Dup, r22Ddnin, b2Dupin] * envs[x2, y2].transfer.r[r22Dup, r22Ddn, r12r2, rb2r2] *
            envs[x2, y2].transfer.b[lb2b, b2Dup, b2Ddn, rb2b] * envs[x2, y2].corner.rb[rb2b, rb2r2]
        @tensor opt = true ψ□ψ[pup1, pup2; pdn1, pdn2] := toppart[pup1, Dupin, l12l2; pdn1, Ddnin, r12r2] * bottompart[pup2, Dupin, l12l2; pdn2, Ddnin, r12r2]
        # ignore_derivatives() do
        #     toppart, bottompart = nothing, nothing
        #     for ii in 1:4
        #         eval(Meta.parse("swgatet$ii, swgateb$ii = nothing, nothing"))
        #     end
        #     GC.gc()
        # end
        @tensor nrm = ψ□ψ[p1, p2, p1, p2]
    else
        error("check input sites")
    end
    # 返回值，自动微分调用此函数时只计算能量(只有一个输入Gate)，且不能返回字典否则无法求导
    if ADflag
        gate = get_op(Gates[1], para)
        @tensor val = ψ□ψ[pup1, pup2, pdn1, pdn2] * gate[pdn1, pdn2, pup1, pup2]
        return val
    else
        vals = Vector{Number}(undef, length(Gates))
        for (ii, tag) in enumerate(Gates)
            gate = get_op(tag, para)
            @tensor tmp = ψ□ψ[pup1, pup2, pdn1, pdn2] * gate[pdn1, pdn2, pup1, pup2]
            vals[ii] = tmp
        end
        rslt = Dict(Gates[ind] => (vals[ind] / nrm) for ind in 1:length(vals))
        return rslt
    end
end


# debug hint:   (1) site1    (2) auxsite
#               (3) auxsite  (4) site2
# or: (1) auxsite    (2) site1
#     (3) site2      (4) auxsite
function _2siteObs_diagSite(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, Gates::Vector{String}, para::Dict{Symbol,Any},
    site1::Vector{Int}, site2::Vector{Int}, get_op::Function; ADflag=false)
    # ADflag ? ipepsbar = ignore_derivatives(ipepsbar) : nothing
    Lx = ipeps.Lx
    Ly = ipeps.Ly
    x1, y1 = site1
    x2, y2 = site2
    M1 = ipeps[x1, y1]
    M1bar = ipepsbar[x1, y1]
    M2 = ipeps[x2, y2]
    M2bar = ipepsbar[x2, y2]
    if x1 == (x2 - 1 - Int(ceil((x2 - 1) / Lx) - 1) * Lx)  # 左上到右下的两个点. 这里调用 CTMRG 求环境的函数
        QuR = get_QuR(ipeps, ipepsbar, envs, x2, y1)  # [lχ, lupD, ldnD; bχ, bupD, bdnD]
        QdL = get_QdL(ipeps, ipepsbar, envs, x1, y2)  # [tχ, tupD, tdnD; rχ, rupD, rdnD]
        gatelt1 = swap_gate(space(M1)[1], space(M1bar)[2]; Eltype=eltype(M1))
        gatelt2 = swap_gate(space(M1)[3], space(M1)[5]; Eltype=eltype(M1))
        gatelt3 = swap_gate(space(gatelt2)[1], space(QuR)[3]; Eltype=eltype(M1))
        gatelt4 = swap_gate(space(gatelt2)[2], space(gatelt3)[2]; Eltype=eltype(M1))
        gatelt5 = swap_gate(space(M1bar)[3], space(M1bar)[4]; Eltype=eltype(M1))
        gatelt6 = swap_gate(space(gatelt5)[1], space(gatelt4)[1]; Eltype=eltype(M1))
        gaterb6 = swap_gate(space(M2)[5], space(M2bar)[4]; Eltype=eltype(M1))
        gaterb5 = swap_gate(space(M2bar)[2], space(M2bar)[3]; Eltype=eltype(M1))
        gaterb4 = swap_gate(space(QdL)[5], space(gaterb5)[2]; Eltype=eltype(M1))
        gaterb3 = swap_gate(space(gaterb4)[1], space(gaterb5)[1]; Eltype=eltype(M1))
        gaterb2 = swap_gate(space(M2)[1], space(M2)[3]; Eltype=eltype(M1))
        gaterb1 = swap_gate(space(QuR)[6], space(gaterb2)[2]; Eltype=eltype(M1))
        @tensor opt = true QuL[(pup1); (pdn1, rχ, rupMD, rdnMD, bχ, bupMD, bdnMD)] :=
            envs[x1, y1].transfer.t[rχin, rχ, bupDin, rupD] * envs[x1, y1].corner.lt[rχin, bχin] *
            envs[x1, y1].transfer.l[bχin, bχ, rupDin, rdnD] * gatelt1[rupDin, rupD, rupDin2, rupDin3] *
            M1[rupDin2, bupDin, pup1in, rupMD, bupMDin] * gatelt2[pup1in2, bupMDin2, pup1in, bupMDin] *
            gatelt3[pup1, rdnMDin2, pup1in2, rdnMD] * gatelt4[bupMDin3, rdnMDin, bupMDin2, rdnMDin2] *
            gatelt5[pdn1in2, rdnMDin, pdn1in, rdnMDin3] * M1bar[rdnD, rupDin3, pdn1in, rdnMDin3, bdnMD] *
            gatelt6[pdn1, bupMD, pdn1in2, bupMDin3]
        @tensor opt = true QdR[(lχ, lupD, ldnD, tχ, tupD, tdnD, pup4); (pdn4)] :=
            envs[x2, y2].corner.rb[lχin, tχin] * envs[x2, y2].transfer.r[lupMDin, ldnDin, tχ, tχin] *
            envs[x2, y2].transfer.b[lχ, tupDin, tdnDin, lχin] * gaterb6[tupDin, ldnDin, tupDin2, ldnDin2] *
            M2bar[ldnD, tdnDin2, pdn4in, ldnDin2, tdnDin] * gaterb5[tdnDin3, pdn4in2, tdnDin2, pdn4in] *
            M2[lupDin3, tupD, pup4in, lupMDin, tupDin2] * gaterb2[lupDin, pup4in2, lupDin3, pup4in] *
            gaterb3[lupDin, tdnDin4, lupDin2, tdnDin3] * gaterb4[lupDin2, pdn4, lupD, pdn4in2] *
            gaterb1[tdnDin4, pup4, tdnD, pup4in2]
        @tensor opt = true ψ□ψ[pup1, pup4; pdn1, pdn4] :=
            QuL[pup1, pdn1, rχ1, rupD1, rdnD1, bχ1, bupD1, bdnD1] * QuR[rχ1, rupD1, rdnD1, bχ2, bupD2, bdnD2] *
            QdL[bχ1, bupD1, bdnD1, rχ3, rupD3, rdnD3] * QdR[rχ3, rupD3, rdnD3, bχ2, bupD2, bdnD2, pup4, pdn4]
        # ignore_derivatives() do
        #     QuL, QuR, QdL, QdR = nothing, nothing, nothing, nothing
        #     for ii in 1:6
        #         eval(Meta.parse("swgatelt$ii, swgaterb$ii = nothing, nothing"))
        #     end
        #     GC.gc()
        # end
        @tensor nrm = ψ□ψ[p1, p2, p1, p2]
    elseif x1 == (x2 + 1 - Int(ceil((x2 + 1) / Lx) - 1) * Lx)  # 右上到左下的两个点.  这里调用 CTMRG 求环境的函数
        QuL = get_QuL(ipeps, ipepsbar, envs, x2, y1)  # [rχ, rupMD, rdnMD, bχ, bupMD, bdnMD]
        QdR = get_QdR(ipeps, ipepsbar, envs, x1, y2)  # [lχ, lupD, ldnD, tχ, tupD, tdnD]
        gatelb1 = swap_gate(space(M2)[1], space(M2bar)[2]; Eltype=eltype(M1))
        gatelb4 = swap_gate(space(M2)[3], space(M2)[5]; Eltype=eltype(M1))
        gatelb2 = swap_gate(space(M2)[4], space(gatelb4)[1]; Eltype=eltype(M1))
        gatelb5 = swap_gate(space(M2bar)[3], space(gatelb4)[2]; Eltype=eltype(M1))
        gatelb3 = swap_gate(space(gatelb2)[1], space(gatelb5)[1]; Eltype=eltype(M1))
        gatelb6 = swap_gate(space(M2bar)[4], space(gatelb5)[2]; Eltype=eltype(M1))
        gatert6 = swap_gate(space(M1)[5], space(M1bar)[4]; Eltype=eltype(M1))
        gatert3 = swap_gate(space(M1bar)[3], space(M1bar)[2]; Eltype=eltype(M1))
        gatert5 = swap_gate(space(M1bar)[1], space(gatert3)[1]; Eltype=eltype(M1))
        gatert2 = swap_gate(space(M1)[3], space(gatert3)[2]; Eltype=eltype(M1))
        gatert1 = swap_gate(space(M1)[1], space(gatert2)[2]; Eltype=eltype(M1))
        gatert4 = swap_gate(space(gatert2)[1], space(gatert5)[1]; Eltype=eltype(M1))
        @tensor opt = true QuR[pup2, pdn2, lχ, lupD, ldnD; bχ, bupD, bdnD] :=
            envs[x1, y1].corner.rt[lχin, bχin] * envs[x1, y1].transfer.t[lχ, lχin, bupDin, bdnDin] *
            envs[x1, y1].transfer.r[lupDin, ldnDin, bχin, bχ] * M1[lupDin2, bupDin, pup2in, lupDin, bupDin2] *
            gatert1[lupD, bdnDin, lupDin2, bdnDin4] * gatert6[bupD, ldnDin, bupDin2, ldnDin2] *
            M1bar[ldnDin3, bdnDin2, pdn2in, ldnDin2, bdnD] * gatert2[pup2in2, bdnDin4, pup2in, bdnDin3] *
            gatert3[pdn2in2, bdnDin3, pdn2in, bdnDin2] * gatert5[ldnDin4, pdn2, ldnDin3, pdn2in2] *
            gatert4[pup2, ldnD, pup2in2, ldnDin4]
        @tensor opt = true QdL[pup3, pdn3, tχ, tupD, tdnD; rχ, rupD, rdnD] :=
            envs[x2, y2].corner.lb[tχin, rχin] * envs[x2, y2].transfer.l[tχ, tχin, rupDin, rdnDin] *
            envs[x2, y2].transfer.b[rχin, tupDin, tdnDin, rχ] * gatelb1[rupDin, tdnD, rupDin2, tdnDin2] *
            M2bar[rdnDin, tdnDin2, pdn3in, rdnDin2, tdnDin] * M2[rupDin2, tupD, pup3in, rupDin3, tupDin2] *
            gatelb4[pup3in2, tupDin3, pup3in, tupDin2] * gatelb2[rupDin4, pup3, rupDin3, pup3in2] *
            gatelb6[rdnD, tupDin, rdnDin2, tupDin4] * gatelb5[pdn3in2, tupDin4, pdn3in, tupDin3] *
            gatelb3[rupD, pdn3, rupDin4, pdn3in2]
        @tensor opt = true ψ□ψ[pup2, pup3; pdn2, pdn3] :=
            QuL[rχ1, rupD1, rdnD1, bχ1, bupD1, bdnD1] * QuR[pup2, pdn2, rχ1, rupD1, rdnD1, bχ2, bupD2, bdnD2] *
            QdL[pup3, pdn3, bχ1, bupD1, bdnD1, rχ3, rupD3, rdnD3] * QdR[rχ3, rupD3, rdnD3, bχ2, bupD2, bdnD2]
        # ignore_derivatives() do
        #     QuL, QuR, QdL, QdR = nothing, nothing, nothing, nothing
        #     for ii in 1:6
        #         eval(Meta.parse("swgatelb$ii, swgatert$ii = nothing, nothing"))
        #     end
        #     GC.gc()
        # end
        @tensor nrm = ψ□ψ[p1, p2, p1, p2]
    else
        error("check input sites")
    end
    # 返回值，自动微分调用此函数时只计算能量(只有一个输入Gate)，且不能返回字典否则无法求导
    if ADflag
        gate = get_op(Gates[1], para)
        @tensor val = ψ□ψ[pup1, pup2, pdn1, pdn2] * gate[pdn1, pdn2, pup1, pup2]
        return val
    else
        vals = Vector{Number}(undef, length(Gates))
        for (ii, tag) in enumerate(Gates)
            gate = get_op(tag, para)
            @tensor tmp = ψ□ψ[pup1, pup2, pdn1, pdn2] * gate[pdn1, pdn2, pup1, pup2]
            vals[ii] = tmp
        end
        rslt = Dict(Gates[ind] => (vals[ind] / nrm) for ind in 1:length(vals))
        return rslt
    end
end

# 内存不友好
# @strided @tensor ψ□ψ[pup1, pup2; pdn1, pdn2] :=
# envs[x1, y1].corner.lt[lt2t1, lt2l] * envs[x1, y1].transfer.l[lt2l, lb2l, l2Dup, l2Ddn] *
# envs[x1, y1].transfer.t[lt2t1, t12t2, t12Dup, t12Ddn] * swgatel1[l2Dup, t12Ddn, l2Dupin, t12Ddnin] *
# M1[l2Dupin, t12Dup, pup1in, Dupin, b12Dupin] * swgatel2[pup1, b12Dupin2, pup1in, b12Dupin] *
# envs[x1, y1].corner.lb[lb2l, lb2b1] * M1bar[l2Ddn, t12Ddnin, pdn1in, Ddnin2, b12Ddn] *
# envs[x1, y1].transfer.b[lb2b1, b12Dup, b12Ddn, b12b2] * swgatel3[pdn1, b12Dupin3, pdn1in, b12Dupin2] *
# swgatel4[b12Dup, Ddnin, b12Dupin3, Ddnin2] *
# swgater1[Dupin, t22Ddn, Dupinin, t22Ddnin3] *
# envs[x2, y2].transfer.t[t12t2, rt2t2, t22Dup, t22Ddn] * M2[Dupinin, t22Dup, pup2in, r2Dup, b22Dupin] *
# swgater2[pup2, t22Ddnin3, pup2in, t22Ddnin2] * swgater3[pdn2, t22Ddnin2, pdn2in, t22Ddnin] *
# M2bar[Ddnin, t22Ddnin, pdn2in, r2Ddnin, b22Ddn] * envs[x2, y2].transfer.b[b12b2, b22Dup, b22Ddn, rb2b2] *
# envs[x2, y2].corner.rt[rt2t2, rt2r] * swgater4[b22Dup, r2Ddn, b22Dupin, r2Ddnin] *
# envs[x2, y2].transfer.r[r2Dup, r2Ddn, rt2r, rb2r] * envs[x2, y2].corner.rb[rb2b2, rb2r]

# 下面这一堆有错误
# @strided @tensor ψ□ψ[pup1, pup2; pdn1, pdn2] :=
# envs[x1, y1].corner.lt[lt2t, lt2l1] * envs[x1, y1].transfer.l[lt2l1, l12l2, l12Dup, l12Ddn] *
# envs[x1, y1].transfer.t[lt2t, rt2t, t2Dup, t2Ddn] * envs[x1, y1].corner.rt[rt2t, rt2r1] *
# swgatet1[l12Dup, t2Ddn, l12Dupin, t2Ddnin] * M1[l12Dupin, t2Dup, pup1in, r12Dupin, Dupin] *
# M1bar[l12Ddn, t2Ddnin, pdn1in, r12Ddnin, Ddnin] * swgatet2[pdn1, r12Ddnin2, pdn1in, r12Ddnin] *
# swgatet3[pup1, r12Ddnin3, pup1in, r12Ddnin2] * swgatet4[r12Dup, r12Ddn, r12Dupin, r12Ddnin3] *
# envs[x1, y1].transfer.r[r12Dup, r12Ddn, rt2r1, r12r2] * envs[x2, y2].transfer.l[l12l2, lb2l2, l22Dup, l22Ddn] *
# envs[x2, y2].corner.lb[lb2l2, lb2b] * swgateb1[Ddnin, l22Dup, Ddninin, l22Dupin3] *
# M2bar[l22Ddn, Ddninin, pdn2in, r22Ddnin, b2Ddn] * swgateb2[pdn2, l22Dupin3, pdn2in, l22Dupin2] *
# swgateb3[l22Dupin2, pup2, l22Dupin, pup2in] * M2[l22Dupin, Dupin, pup2in, r22Dup, b2Dupin] *
# swgateb4[r22Ddn, b2Dup, r22Ddnin, b2Dupin] * envs[x2, y2].transfer.r[r22Dup, r22Ddn, r12r2, rb2r2] *
# envs[x2, y2].transfer.b[lb2b, b2Dup, b2Ddn, rb2b] * envs[x2, y2].corner.rb[rb2b, rb2r2]