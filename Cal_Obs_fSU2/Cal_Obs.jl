function Cal_Obs_1site(ipeps::iPEPS, envs::iPEPSenv, Ops::Vector{Symbol}, para::Dict{Symbol,Any}; site=[1, 1])
    x = site[1]
    y = site[2]
    vals = Vector{Number}(undef, length(Ops))
    # 顺序：CTTM?̄MCCTCT
    @tensor contractcheck = true ψ□ψ[pup, pdn] :=
        envs[x, y].corner.lt[toT, toL] * envs[x, y].transfer.t[toT, toRT, toMtup, toMtdn] *
        envs[x, y].transfer.l[toL, toLB, toMlup, toMldn] * ipeps[x, y][toMlup, toMtup, pup, toMrup, toMbup] *
        ipeps[x, y]'[toMrdn, toMbdn, toMldn, toMtdn, pdn] * envs[x, y].corner.rt[toRT, toR] *
        envs[x, y].corner.lb[toLB, toB] * envs[x, y].transfer.r[toMrup, toMrdn, toR, toRB] *
        envs[x, y].transfer.b[toB, toMbup, toMbdn, btoRB] * envs[x, y].corner.rb[btoRB, toRB]
    @tensor nrm = ψ□ψ[p, p]
    for (ii, tag) in enumerate(Ops)
        op = get_op(tag, para)
        @tensor tmp = ψ□ψ[pup, pdn] * op[pup, pdn]
        vals[ii] = scalar(tmp)
    end
    rslt = Dict(Ops[ind] => (vals[ind] / nrm) for ind in 1:length(vals))
    return rslt
end


function Cal_Obs_2site(ipeps::iPEPS, envs::iPEPSenv, Gates::Vector{String}, para::Dict{Symbol,Any};
    site1::Vector{Int}, site2::Vector{Int}, get_op::Function)
    Lx = ipeps.Lx
    Ly = ipeps.Ly
    @assert Lx >= 2 || Ly >= 2
    @assert 0.99 < norm(site1 - site2) < 1.01  # 近邻格点
    if sum(site1) > sum(site2)
        site1, site2 = site2, site1
    end
    vals = Vector{Number}(undef, length(Gates))
    x1, y1 = site1
    x2, y2 = site2
    if x2 == x1 + 1  # 横向的两个点.  顺序：CTTMMbarCTTMMbarCTTC.  这里要小心fSU2是否能正确处理交换门？
        @tensor contractcheck = true ψ□ψ[pup1, pup2; pdn1, pdn2] :=
            envs[x1, y1].corner.lt[lt2t1, lt2l] * envs[x1, y1].transfer.l[lt2l, lb2l, l2Dup, l2Ddn] *
            envs[x1, y1].transfer.t[lt2t1, t12t2, t12Dup, t12Ddn] * ipeps[x1, y1][l2Dup, t12Dup, pup1, Dupin, b12Dup] *
            ipeps[x1, y1]'[Ddnin, b12Ddn, l2Ddn, t12Ddn, pdn1] * envs[x1, y1].corner.lb[lb2l, lb2b1] *
            envs[x1, y1].transfer.b[lb2b1, b12Dup, b12Ddn, b12b2] * envs[x2, y2].transfer.t[t12t2, rt2t2, t22Dup, t22Ddn] *
            ipeps[x2, y2][Dupin, t22Dup, pup2, r2Dup, b22Dup] * ipeps[x2, y2]'[r2Ddn, b22Ddn, Ddnin, t22Ddn, pdn2] *
            envs[x2, y2].corner.rt[rt2t2, rt2r] * envs[x2, y2].transfer.r[r2Dup, r2Ddn, rt2r, rb2r] *
            envs[x2, y2].transfer.b[b12b2, b22Dup, b22Ddn, rb2b2] * envs[x2, y2].corner.rb[rb2b2, rb2r]
        @tensor nrm = ψ□ψ[p1, p2, p1, p2]
    elseif y2 == y1 + 1  # 纵向的两个点
        @tensor contractcheck = true ψ□ψ[pup1, pup2; pdn1, pdn2] :=
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
        @tensor contractcheck = true tmp[] := ψ□ψ[pup1, pup2, pdn1, pdn2] * gate[pdn1, pdn2, pup1, pup2]
        vals[ii] = scalar(tmp)
    end
    rslt = Dict(Gates[ind] => (vals[ind] / nrm) for ind in 1:length(vals))
    return rslt
end