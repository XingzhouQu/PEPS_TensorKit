using LinearAlgebra

function Cal_Obs_1site(ipeps::iPEPS, envs::iPEPSenv, Ops::Vector{Symbol}; site=[1, 1])
    x = site[1]
    y = site[2]
    vals = Vector{Number}(undef, length(Ops))
    # 顺序：CTTM?̄MCCTCT
    @tensor ψ□ψ[pup, pdn] :=
        envs[x, y].corner.lt[toT, toL] * envs[x, y].transfer.t[toT, toRT, toMtup, toMtdn] *
        envs[x, y].transfer.l[toL, toLB, toMlup, toMldn] * ipeps[x, y][toMlup, toMtup, pup, toMrup, toMbup] *
        ipeps[x, y]'[toMrdn, toMbdn, toMldn, toMtdn, pdn] * envs[x, y].corner.rt[toRT, toR] *
        envs[x, y].corner.lb[toLB, toB] * envs[x, y].transfer.r[toMrup, toMrdn, toR, toRB] *
        envs[x, y].transfer.b[toB, toMbup, toMbdn, btoRB] * envs[x, y].corner.rb[btoRB, toRB]
    @tensor nrm = ψ□ψ[p, p]
    for (ii, sym) in enunerate(Ops)
        op = get_op(sym)
        @tensor vals[ii] = ψ□ψ[pup, pdn] * op[pup, pdn]
    end
    rslt = Dict(Ops[ind] => (vals[ind] / nrm) for ind in 1:length(vals))
    return rslt
end


function Cal_Obs_2site(ipeps::iPEPS, envs::iPEPSenv, Ops::Vector{Symbol}; site1=[1, 1], site2=[1, 2])
    @assert Lx >= 2 || Ly >= 2
    @assert 0.99 < norm(site1 - site2) < 1.01  # 近邻格点
    sum(site1) < sum(site2) ? nothing : site1, site2 = site2, site1
    vals = Vector{Number}(undef, length(Ops))
    x1, y1 = site1
    x2, y2 = site2
    if y2 == y1 + 1  # 横向的两个点.  顺序：CTTMMbarCTTMMbarCTTC
        @tensor ψ□ψ[pup1, pup2; pdn1, pdn2] :=
            envs[x1, y1].corner.lt[] * envs[x1, y1].transfer.l[] *
            envs[x1, y1].transfer.t[] * ipeps[x1, y1][] *
            ipeps[x1, y1]'[] * envs[x1, y1].corner.lb[] *
            envs[x1, y1].transfer.b[] * envs[x2, y2].transfer.t[] *
            ipeps[x2, y2][] * ipeps[x2, y2]'[] *
            envs[x2, y2].corner.rt[] * envs[x2, y2].transfer.r[] *
            envs[x2, y2].transfer.b[] * envs[x2, y2].corner.rb[]
        @tensor nrm = ψ□ψ[]
    elseif x2 == x1 + 1  # 纵向的两个点
        @tensor ψ□ψ[pup1, pup2; pdn1, pdn2] :=
            envs[x1, y1].corner.lt[] * envs[x1, y1].transfer.l[] *
            envs[x1, y1].transfer.t[] * ipeps[x1, y1][] *
            ipeps[x1, y1]'[] * envs[x1, y1].corner.rt[] *
            envs[x1, y1].transfer.r[] * envs[x2, y2].transfer.l[] *
            ipeps[x2, y2][] * ipeps[x2, y2]'[] *
            envs[x2, y2].transfer.r[] * envs[x2, y2].corner.lb[] *
            envs[x2, y2].transfer.b[] * envs[x2, y2].corner.rb[]
        @tensor nrm = ψ□ψ[]
    else
        error("check input sites")
    end
    for (ii, sym) in enunerate(Ops)
        op = get_op(sym)
        @tensor vals[ii] = ψ□ψ[pup1, pdn1, pup2, pdn2] * op[pup1, pdn1, pup2, pdn2]
    end
    rslt = Dict(Ops[ind] => (vals[ind] / nrm) for ind in 1:length(vals))
    return rslt
end