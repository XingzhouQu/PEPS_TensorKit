function Cal_Obs_1site(ipeps::iPEPS, envs::iPEPSenv, Ops::Vector{String}, para::Dict{Symbol,Any}; site=[1, 1], get_op::Function)
    x = site[1]
    y = site[2]
    vals = Vector{Number}(undef, length(Ops))
    # 顺序：CTTM?̄MCCTCT
    @tensor contractcheck = false ψ□ψ[pup, pdn] :=
        envs[x, y].corner.lt[toT, toL] * envs[x, y].transfer.t[toT, toRT, toMtup, toMtdn] *
        envs[x, y].transfer.l[toL, toLB, toMlup, toMldn] * ipeps[x, y][toMlup, toMtup, pup, toMrup, toMbup] *
        ipeps[x, y]'[toMrdn, toMbdn, toMldn, toMtdn, pdn] * envs[x, y].corner.rt[toRT, toR] *
        envs[x, y].corner.lb[toLB, toB] * envs[x, y].transfer.r[toMrup, toMrdn, toR, toRB] *
        envs[x, y].transfer.b[toB, toMbup, toMbdn, btoRB] * envs[x, y].corner.rb[btoRB, toRB]
    @tensor nrm = ψ□ψ[p, p]
    for (ii, tag) in enumerate(Ops)
        op = get_op(tag, para)
        @tensor contractcheck = false tmp[] := ψ□ψ[pup, pdn] * op[pdn, pup]
        vals[ii] = scalar(tmp)
    end
    rslt = Dict(Ops[ind] => (vals[ind] / nrm) for ind in 1:length(vals))
    return rslt
end

"""
计算两点观测量。输入的site convention:\n
[site1, site2] = \n
[左,右]; [上,下]; [左上,右下]; [右上,左下]
"""
function Cal_Obs_2site(ipeps::iPEPS, envs::iPEPSenv, Gates::Vector{String}, para::Dict{Symbol,Any};
    site1::Vector{Int}, site2::Vector{Int}, get_op::Function)
    Lx = ipeps.Lx
    Ly = ipeps.Ly
    @assert Lx >= 2 || Ly >= 2 "The unit cell should contain no less than two sites"
    if 0.99 < norm(site1 - site2) < 1.01  # 近邻格点
        println("Calculating $Gates at site $site1 and site $site2")
        rslt = _2siteObs_adjSite(ipeps, envs, Gates, para, site1, site2, get_op)
        return rslt
    elseif (abs(site1[1] - site2[1]) == 1) && (abs(site1[2] - site2[2]) == 1)
        println("Calculating $Gates at site $site1 and site $site2")
        rslt = _2siteObs_diagSite(ipeps, envs, Gates, para, site1, site2, get_op)
        return rslt
    else
        error("Larger distance is not supported yet.")
    end
end


function _2siteObs_adjSite(ipeps::iPEPS, envs::iPEPSenv, Gates::Vector{String}, para::Dict{Symbol,Any},
    site1::Vector{Int}, site2::Vector{Int}, get_op::Function)
    vals = Vector{Number}(undef, length(Gates))
    x1, y1 = site1
    x2, y2 = site2
    if y2 == y1  # 横向的两个点.  顺序：CTTMMbarCTTMMbarCTTC.
        @tensor contractcheck = false ψ□ψ[pup1, pup2; pdn1, pdn2] :=
            envs[x1, y1].corner.lt[lt2t1, lt2l] * envs[x1, y1].transfer.l[lt2l, lb2l, l2Dup, l2Ddn] *
            envs[x1, y1].transfer.t[lt2t1, t12t2, t12Dup, t12Ddn] * ipeps[x1, y1][l2Dup, t12Dup, pup1, Dupin, b12Dup] *
            ipeps[x1, y1]'[Ddnin, b12Ddn, l2Ddn, t12Ddn, pdn1] * envs[x1, y1].corner.lb[lb2l, lb2b1] *
            envs[x1, y1].transfer.b[lb2b1, b12Dup, b12Ddn, b12b2] * envs[x2, y2].transfer.t[t12t2, rt2t2, t22Dup, t22Ddn] *
            ipeps[x2, y2][Dupin, t22Dup, pup2, r2Dup, b22Dup] * ipeps[x2, y2]'[r2Ddn, b22Ddn, Ddnin, t22Ddn, pdn2] *
            envs[x2, y2].corner.rt[rt2t2, rt2r] * envs[x2, y2].transfer.r[r2Dup, r2Ddn, rt2r, rb2r] *
            envs[x2, y2].transfer.b[b12b2, b22Dup, b22Ddn, rb2b2] * envs[x2, y2].corner.rb[rb2b2, rb2r]
        @tensor nrm = ψ□ψ[p1, p2, p1, p2]
    elseif x2 == x1  # 纵向的两个点
        @tensor contractcheck = false ψ□ψ[pup1, pup2; pdn1, pdn2] :=
            envs[x1, y1].corner.lt[lt2t, lt2l1] * envs[x1, y1].transfer.l[lt2l1, l12l2, l12Dup, l12Ddn] *
            envs[x1, y1].transfer.t[lt2t, rt2t, t2Dup, t2Ddn] * ipeps[x1, y1][l12Dup, t2Dup, pup1, r12Dup, Dupin] *
            ipeps[x1, y1]'[r12Ddn, Ddnin, l12Ddn, t2Ddn, pdn1] * envs[x1, y1].corner.rt[rt2t, rt2r1] *
            envs[x1, y1].transfer.r[r12Dup, r12Ddn, rt2r1, r12r2] * envs[x2, y2].transfer.l[l12l2, lb2l2, l22Dup, l22Ddn] *
            ipeps[x2, y2][l22Dup, Dupin, pup2, r22Dup, b2Dup] * ipeps[x2, y2]'[r22Ddn, b2Ddn, l22Ddn, Ddnin, pdn2] *
            envs[x2, y2].transfer.r[r22Dup, r22Ddn, r12r2, rb2r2] * envs[x2, y2].corner.lb[lb2l2, lb2b] *
            envs[x2, y2].transfer.b[lb2b, b2Dup, b2Ddn, rb2b] * envs[x2, y2].corner.rb[rb2b, rb2r2]
        @tensor nrm = ψ□ψ[p1, p2, p1, p2]
    else
        error("check input sites")
    end
    for (ii, tag) in enumerate(Gates)
        gate = get_op(tag, para)
        @tensor contractcheck = false tmp[] := ψ□ψ[pup1, pup2, pdn1, pdn2] * gate[pdn1, pdn2, pup1, pup2]
        vals[ii] = scalar(tmp)
    end
    rslt = Dict(Gates[ind] => (vals[ind] / nrm) for ind in 1:length(vals))
    return rslt
end


# debug hint:   (1) site1    (2) auxsite
#               (3) auxsite  (4) site2
# or: (1) auxsite    (2) site1
#     (3) site2      (4) auxsite
function _2siteObs_diagSite(ipeps::iPEPS, envs::iPEPSenv, Gates::Vector{String}, para::Dict{Symbol,Any},
    site1::Vector{Int}, site2::Vector{Int}, get_op::Function)
    Lx = ipeps.Lx
    Ly = ipeps.Ly
    vals = Vector{Number}(undef, length(Gates))
    x1, y1 = site1
    x2, y2 = site2
    if x1 == (x2 - 1 - Int(ceil((x2 - 1) / Lx) - 1) * Lx)  # 左上到右下的两个点.  顺序：CTTMMbarTTCMMbarMMbarTTMMbarTTC.
        @tensor contractcheck = true ψ□ψ[pup1, pup4; pdn1, pdn4] :=
            envs[x1, y1].corner.lt[CLTr1, CLTb1] * envs[x1, y1].transfer.t[CLTr1, TTr1, TTupD1, TTdnD1] *
            envs[x1, y1].transfer.l[CLTb1, TLb1, TLupD1, TLdnD1] * ipeps[x1, y1][TLupD1, TTupD1, pup1, rupD1, bupD1] *
            ipeps[x1, y1]'[rdnD1, bdnD1, TLdnD1, TTdnD1, pdn1] * envs[x2, y1].transfer.t[TTr1, TTr2, TTupD2, TTdnD2] *
            envs[x1, y2].transfer.l[TLb1, TLb3, TLupD3, TLdnD3] * envs[x2, y1].corner.rt[TTr2, CRTb2] *
            envs[x1, y2].corner.lb[TLb3, CLBr3] * ipeps[x2, y1][rupD1, TTupD2, p2, rupD2, bupD2] *
            ipeps[x2, y1]'[rdnD2, bdnD2, rdnD1, TTdnD2, p2] * ipeps[x1, y2][TLupD3, bupD1, p3, rupD3, bupD3] *
            ipeps[x1, y2]'[rdnD3, bdnD3, TLdnD3, bdnD1, p3] * envs[x2, y1].transfer.r[rupD2, rdnD2, CRTb2, TRb2] *
            envs[x1, y2].transfer.b[CLBr3, bupD3, bdnD3, TBr3] * ipeps[x2, y2][rupD3, bupD2, pup4, rupD4, bupD4] *
            ipeps[x2, y2]'[rdnD4, bdnD4, rdnD3, bdnD2, pdn4] * envs[x2, y2].transfer.r[rupD4, rdnD4, TRb2, TRb4] *
            envs[x2, y2].transfer.b[TBr3, bupD4, bdnD4, TBr4] * envs[x2, y2].corner.rb[TBr4, TRb4]
        @tensor nrm = ψ□ψ[p1, p2, p1, p2]
    elseif x1 == (x2 + 1 - Int(ceil((x2 + 1) / Lx) - 1) * Lx)  # 右上到左下的两个点
        @tensor contractcheck = true ψ□ψ[pup2, pup3; pdn2, pdn3] :=
            envs[x2, y1].corner.lt[CLTr1, CLTb1] * envs[x2, y1].transfer.t[CLTr1, TTr1, TTupD1, TTdnD1] *
            envs[x2, y1].transfer.l[CLTb1, TLb1, TLupD1, TLdnD1] * ipeps[x2, y1][TLupD1, TTupD1, p1, rupD1, bupD1] *
            ipeps[x2, y1]'[rdnD1, bdnD1, TLdnD1, TTdnD1, p1] * envs[x1, y1].transfer.t[TTr1, TTr2, TTupD2, TTdnD2] *
            envs[x2, y2].transfer.l[TLb1, TLb3, TLupD3, TLdnD3] * envs[x1, y1].corner.rt[TTr2, CRTb2] *
            envs[x2, y2].corner.lb[TLb3, CLBr3] * ipeps[x1, y1][rupD1, TTupD2, pup2, rupD2, bupD2] *
            ipeps[x1, y1]'[rdnD2, bdnD2, rdnD1, TTdnD2, pdn2] * ipeps[x2, y2][TLupD3, bupD1, pup3, rupD3, bupD3] *
            ipeps[x2, y2]'[rdnD3, bdnD3, TLdnD3, bdnD1, pdn3] * envs[x1, y1].transfer.r[rupD2, rdnD2, CRTb2, TRb2] *
            envs[x2, y2].transfer.b[CLBr3, bupD3, bdnD3, TBr3] * ipeps[x1, y2][rupD3, bupD2, p4, rupD4, bupD4] *
            ipeps[x1, y2]'[rdnD4, bdnD4, rdnD3, bdnD2, p4] * envs[x1, y2].transfer.r[rupD4, rdnD4, TRb2, TRb4] *
            envs[x1, y2].transfer.b[TBr3, bupD4, bdnD4, TBr4] * envs[x1, y2].corner.rb[TBr4, TRb4]
        @tensor nrm = ψ□ψ[p1, p2, p1, p2]
    else
        error("check input sites")
    end
    for (ii, tag) in enumerate(Gates)
        gate = get_op(tag, para)
        @tensor contractcheck = false tmp[] := ψ□ψ[pup1, pup2, pdn1, pdn2] * gate[pdn1, pdn2, pup1, pup2]
        vals[ii] = scalar(tmp)
    end
    rslt = Dict(Gates[ind] => (vals[ind] / nrm) for ind in 1:length(vals))
    return rslt
end