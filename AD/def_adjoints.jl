using ChainRulesCore


ChainRulesCore.@non_differentiable TensorKit.SortedVectorDict()
"""
Custom Gradient: Use Zygote.@adjoint to specify how to compute the gradient of f(x) = x^3 manually.

Zygote.@adjoint f(x) = f(x), Δ -> (3x^2 * Δ,)

The first part, f(x), specifies the value of the function.

The second part, Δ -> (3x^2 * Δ,), specifies how to compute the gradient. 
Δ represents the upstream gradient (the gradient of the loss with respect to the output of f).
"""

# 这里要确保梯度流 Δ 都是 TensorMap 类型？
"""
Trying to define basic gradient rules for local obs functions here

    since ∂Tr(XᵀBX)/∂X = BX + BᵀX, for an Hermitian operator ô = ô⁺ and real wave function ψ, 
    we have ∂Tr(ψ⁺ôψ)/∂ψ = 2ôψ
"""
function ChainRulesCore.rrule(::typeof(_2siteObs_diagSite), ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, Gates::Vector{String}, para::Dict{Symbol,Any}, site1::Vector{Int}, site2::Vector{Int}, get_op::Function; ADflag=true)
    # args = ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, Gates::Vector{String}, para::Dict{Symbol,Any}, site1::Vector{Int}, site2::Vector{Int}, get_op::Function
    y = _2siteObs_diagSite(ipeps, ipepsbar, envs, Gates, para, site1, site2, get_op; ADflag=true)
    function diagObsPullback(ȳ)
        f̄ = NoTangent()  # The function Cal_Obs has no fields (i.e. it is not a closure) and can not be perturbed. Therefore its tangent (f̄) is a NoTangent.
        # The struct ipeps::iPEPS gets a Tangent{iPEPS} structural tangent, which stores the tangents of fields of ipeps.
        #     The tangent of the field Ms is specified,
        #     The tangent of the field Lx and Ly are NoTangent(), because they can not be perturbed, either.
        ipeps̄ = Tangent{iPEPS}(; Ms = ȳ * MsTangent_diag(ipeps, ipepsbar, envs, Gates, para, site1, site2, get_op; ADflag=true), 
                Lx = NoTangent(), Ly = NoTangent()) |> canonicalize
        # envs̄ = Tangent{iPEPSenv}(; Envs=EnvsTangent_diag(args...), Lx=NoTangent(), Ly=NoTangent())
        
        return f̄, ipeps̄
    end
    return y, diagObsPullback
end


function ChainRulesCore.rrule(::typeof(_2siteObs_adjSite), ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, Gates::Vector{String}, para::Dict{Symbol,Any}, site1::Vector{Int}, site2::Vector{Int}, get_op::Function; ADflag=true)
    # args = ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, Gates::Vector{String}, para::Dict{Symbol,Any}, site1::Vector{Int}, site2::Vector{Int}, get_op::Function
    y = _2siteObs_adjSite(ipeps, ipepsbar, envs, Gates, para, site1, site2, get_op; ADflag=true)
    function adjObsPullback(ȳ)
        f̄ = NoTangent()  # The function Cal_Obs has no fields (i.e. it is not a closure) and can not be perturbed. Therefore its tangent (f̄) is a NoTangent.
        # The struct ipeps::iPEPS gets a Tangent{iPEPS} structural tangent, which stores the tangents of fields of ipeps.
        #     The tangent of the field Ms is specified,
        #     The tangent of the field Lx and Ly are NoTangent(), because they can not be perturbed, either.
        ipeps̄ = iPEPS(ȳ .* MsTangent_adj(ipeps, ipepsbar, envs, Gates, para, site1, site2, get_op; ADflag=true), Lx, Ly)
        @show typeof(ipeps̄)
        @show typeof(ȳ)
        # envs̄ = Tangent{iPEPSenv}(; Envs=EnvsTangent_adj(args...), Lx=NoTangent(), Ly=NoTangent())
        
        return f̄, ipeps̄
    end
    return y, adjObsPullback
end

function MsTangent_diag(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, Gates::Vector{String}, para::Dict{Symbol,Any}, site1::Vector{Int}, site2::Vector{Int}, get_op::Function; ADflag=true)
    ADflag ? ipepsbar = ignore_derivatives(ipepsbar) : nothing
    Lx = ipeps.Lx
    x1, y1 = site1
    x2, y2 = site2
    M1 = ipeps[x1, y1]
    M1bar = ipepsbar[x1, y1]
    M2 = ipeps[x2, y2]
    M2bar = ipepsbar[x2, y2]
    gate = get_op(Gates[1], para)
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
        # 对两个 ipeps tensor 的梯度.   codomain 是 M1 指标, domain 是 M2 指标
        # M1[rupDin2, bupDin, pup1in, rupD1, bupMDin],  M2[lupDin3, bupD2, pup4in, lupMDin, tupDin2]
        @tensoropt ψ⁺ô□[(rupDin2, bupDin, pup1in, rupD1, bupMDin); (lupDin3, bupD2, pup4in, lupMDin, tupDin2)] := 
            envs[x1, y1].transfer.t[rχin, rχ1, bupDin, rupD] * envs[x1, y1].corner.lt[rχin, bχin] *
            envs[x1, y1].transfer.l[bχin, bχ1, rupDin, rdnD] * gatelt1[rupDin, rupD, rupDin2, rupDin3] *
            gatelt2[pup1in2, bupMDin2, pup1in, bupMDin] * gatelt3[pup1, rdnMDin2, pup1in2, rdnD1] * 
            gatelt4[bupMDin3, rdnMDin, bupMDin2, rdnMDin2] * gatelt5[pdn1in2, rdnMDin, pdn1in, rdnMDin3] * 
            M1bar[rdnD, rupDin3, pdn1in, rdnMDin3, bdnD1] * gatelt6[pdn1, bupD1, pdn1in2, bupMDin3] *  # QuL \ M1
            QuR[rχ1, rupD1, rdnD1, bχ2, bupD2, bdnD2] *  # QuR
            QdL[bχ1, bupD1, bdnD1, rχ3, rupD3, rdnD3] *  # QdL
            envs[x2, y2].corner.rb[lχin, tχin] * envs[x2, y2].transfer.r[lupMDin, ldnDin, bχ2, tχin] *
            envs[x2, y2].transfer.b[rχ3, tupDin, tdnDin, lχin] * gaterb6[tupDin, ldnDin, tupDin2, ldnDin2] *
            M2bar[rdnD3, tdnDin2, pdn4in, ldnDin2, tdnDin] * gaterb5[tdnDin3, pdn4in2, tdnDin2, pdn4in] *
            gaterb2[lupDin, pup4in2, lupDin3, pup4in] * gaterb3[lupDin, tdnDin4, lupDin2, tdnDin3] * 
            gaterb4[lupDin2, pdn4, rupD3, pdn4in2] * gaterb1[tdnDin4, pup4, bdnD2, pup4in2] *   # QdR \ M2
            gate[pdn1, pdn4, pup1, pup4]
        @tensoropt M1grd[rupDin2, bupDin, pup1in; rupD1, bupMDin] := 
            ψ⁺ô□[rupDin2, bupDin, pup1in, rupD1, bupMDin, lupDin3, bupD2, pup4in, lupMDin, tupDin2] * 
            M2[lupDin3, bupD2, pup4in, lupMDin, tupDin2]
        @tensoropt M2grd[lupDin3, bupD2, pup4in; lupMDin, tupDin2] :=
            ψ⁺ô□[rupDin2, bupDin, pup1in, rupD1, bupMDin, lupDin3, bupD2, pup4in, lupMDin, tupDin2] *
            M1[rupDin2, bupDin, pup1in, rupD1, bupMDin]
        @tensoropt nrm = M1grd[rupDin2, bupDin, pup1in, rupD1, bupMDin] * M1[rupDin2, bupDin, pup1in; rupD1, bupMDin]
    elseif x1 == (x2 + 1 - Int(ceil((x2 + 1) / Lx) - 1) * Lx)  # 右上到左下的两个点.
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
        @tensoropt ψ⁺ô□[(lupDin2, bupDin, pup2in, lupDin, bupDin2); (rupDin2, bupD1, pup3in, rupDin3, tupDin2)] :=
            QuL[rχ1, rupD1, rdnD1, bχ1, bupD1, bdnD1] * 
            envs[x1, y1].corner.rt[lχin, bχin] * envs[x1, y1].transfer.t[rχ1, lχin, bupDin, bdnDin] *
            envs[x1, y1].transfer.r[lupDin, ldnDin, bχin, bχ2] * 
            gatert1[rupD1, bdnDin, lupDin2, bdnDin4] * gatert6[bupD2, ldnDin, bupDin2, ldnDin2] *
            M1bar[ldnDin3, bdnDin2, pdn2in, ldnDin2, bdnD2] * gatert2[pup2in2, bdnDin4, pup2in, bdnDin3] *
            gatert3[pdn2in2, bdnDin3, pdn2in, bdnDin2] * gatert5[ldnDin4, pdn2, ldnDin3, pdn2in2] *
            gatert4[pup2, rdnD1, pup2in2, ldnDin4] * # QuR
            envs[x2, y2].corner.lb[tχin, rχin] * envs[x2, y2].transfer.l[bχ1, tχin, rupDin, rdnDin] *
            envs[x2, y2].transfer.b[rχin, tupDin, tdnDin, rχ3] * gatelb1[rupDin, bdnD1, rupDin2, tdnDin2] *
            M2bar[rdnDin, tdnDin2, pdn3in, rdnDin2, tdnDin] * 
            gatelb4[pup3in2, tupDin3, pup3in, tupDin2] * gatelb2[rupDin4, pup3, rupDin3, pup3in2] *
            gatelb6[rdnD3, tupDin, rdnDin2, tupDin4] * gatelb5[pdn3in2, tupDin4, pdn3in, tupDin3] *
            gatelb3[rupD3, pdn3, rupDin4, pdn3in2] * # QdL
            QdR[rχ3, rupD3, rdnD3, bχ2, bupD2, bdnD2] * gate[pdn2, pdn3, pup2, pup3]
        @tensoropt M1grd[lupDin2, bupDin, pup2in; lupDin, bupDin2] := 
            ψ⁺ô□[lupDin2, bupDin, pup2in, lupDin, bupDin2, rupDin2, bupD1, pup3in, rupDin3, tupDin2] * 
            M2[rupDin2, bupD1, pup3in, rupDin3, tupDin2]
        @tensoropt M2grd[rupDin2, bupD1, pup3in; rupDin3, tupDin2] := 
            ψ⁺ô□[lupDin2, bupDin, pup2in, lupDin, bupDin2, rupDin2, bupD1, pup3in, rupDin3, tupDin2] * 
            M1[lupDin2, bupDin, pup2in, lupDin, bupDin2]
        @tensoropt nrm = M1grd[lupDin2, bupDin, pup2in, lupDin, bupDin2] * M1[lupDin2, bupDin, pup2in; lupDin, bupDin2]
    else
        error("check input sites")
    end
    ipepsNew = deepcopy(ipeps)
    ipepsNew[x1, y1] = 2.0 * M1grd / nrm
    ipepsNew[x2, y2] = 2.0 * M2grd / nrm
    return ipepsNew.Ms
end

function MsTangent_adj(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, Gates::Vector{String}, para::Dict{Symbol,Any}, site1::Vector{Int}, site2::Vector{Int}, get_op::Function; ADflag=true)
    ADflag ? ipepsbar = ignore_derivatives(ipepsbar) : nothing
    x1, y1 = site1
    x2, y2 = site2
    M1 = ipeps[x1, y1]
    M1bar = ipepsbar[x1, y1]
    M2 = ipeps[x2, y2]
    M2bar = ipepsbar[x2, y2]
    gate = get_op(Gates[1], para)
    if y2 == y1  # 横向的两个点.
        swgatel1 = swap_gate(space(M1)[1], space(M1bar)[2]; Eltype=eltype(M1))
        swgatel2 = swap_gate(space(M1)[3], space(M1)[5]; Eltype=eltype(M1))
        swgatel3 = swap_gate(space(M1bar)[3], space(swgatel2)[2]; Eltype=eltype(M1))
        swgatel4 = swap_gate(space(swgatel3)[2], space(M1bar)[4]; Eltype=eltype(M1))
        swgater4 = swap_gate(space(M2)[5], space(M2bar)[4]; Eltype=eltype(M1))
        swgater3 = swap_gate(space(M2bar)[3], space(M2bar)[2]; Eltype=eltype(M1))
        swgater2 = swap_gate(space(M2)[3], space(swgater3)[2]; Eltype=eltype(M1))
        swgater1 = swap_gate(space(M2)[1], space(swgater2)[2]; Eltype=eltype(M1))
        # codomain(左侧)是M1指标，domain(右侧)是M2指标
        @tensoropt ψ⁺ô□[(l2Dupin, t12Dup, pup1in, Dupin, b12Dupin); (Dupinin, t22Dup, pup2in, r2Dup, b22Dupin)] := 
            envs[x1, y1].corner.lt[lt2t1, lt2l] * envs[x1, y1].transfer.t[lt2t1, t12t2, t12Dup, t12Ddn] *
            swgatel1[l2Dup, t12Ddn, l2Dupin, t12Ddnin] * swgatel2[pup1, b12Dupin2, pup1in, b12Dupin] * 
            swgatel3[pdn1, b12Dupin3, pdn1in, b12Dupin2] * M1bar[l2Ddn, t12Ddnin, pdn1in, Ddnin2, b12Ddn] * 
            envs[x1, y1].transfer.l[lt2l, lb2l, l2Dup, l2Ddn] * swgatel4[b12Dup, Ddnin, b12Dupin3, Ddnin2] * 
            envs[x1, y1].transfer.b[lb2b1, b12Dup, b12Ddn, b12b2] * envs[x1, y1].corner.lb[lb2l, lb2b1] *  # leftpart
            swgater1[Dupin, t22Ddn, Dupinin, t22Ddnin3] * envs[x2, y2].transfer.t[t12t2, rt2t2, t22Dup, t22Ddn] * 
            swgater2[pup2, t22Ddnin3, pup2in, t22Ddnin2] * swgater3[pdn2, t22Ddnin2, pdn2in, t22Ddnin] *
            M2bar[Ddnin, t22Ddnin, pdn2in, r2Ddnin, b22Ddn] * envs[x2, y2].transfer.b[b12b2, b22Dup, b22Ddn, rb2b2] *
            envs[x2, y2].corner.rt[rt2t2, rt2r] * swgater4[b22Dup, r2Ddn, b22Dupin, r2Ddnin] *
            envs[x2, y2].transfer.r[r2Dup, r2Ddn, rt2r, rb2r] * envs[x2, y2].corner.rb[rb2b2, rb2r] * # rightpart
            gate[pdn1, pdn2, pup1, pup2]
        @tensoropt M1grd[l2Dupin, t12Dup, pup1in; Dupin, b12Dupin] := 
            ψ⁺ô□[l2Dupin, t12Dup, pup1in, Dupin, b12Dupin, Dupinin, t22Dup, pup2in, r2Dup, b22Dupin] * 
            M2[Dupinin, t22Dup, pup2in, r2Dup, b22Dupin]
        @tensoropt M2grd[Dupinin, t22Dup, pup2in; r2Dup, b22Dupin] :=
            ψ⁺ô□[l2Dupin, t12Dup, pup1in, Dupin, b12Dupin, Dupinin, t22Dup, pup2in, r2Dup, b22Dupin] *
            M1[l2Dupin, t12Dup, pup1in, Dupin, b12Dupin]
        @tensoropt nrm = M1grd[l2Dupin, t12Dup, pup1in, Dupin, b12Dupin] * M1[l2Dupin, t12Dup, pup1in, Dupin, b12Dupin]
    elseif x2 == x1  # 纵向的两个点
        swgatet1 = swap_gate(space(M1)[1], space(M1bar)[2]; Eltype=eltype(M1))
        swgatet2 = swap_gate(space(M1bar)[3], space(M1bar)[4]; Eltype=eltype(M1))
        swgatet3 = swap_gate(space(M1)[3], space(swgatet2)[2]; Eltype=eltype(M1))
        swgatet4 = swap_gate(space(M1)[5], space(swgatet3)[2]; Eltype=eltype(M1))
        swgateb4 = swap_gate(space(M2bar)[4], space(M2)[5]; Eltype=eltype(M1))
        swgateb3 = swap_gate(space(M2)[1], space(M2)[3]; Eltype=eltype(M1))
        swgateb2 = swap_gate(space(M2bar)[3], space(swgateb3)[1]; Eltype=eltype(M1))
        swgateb1 = swap_gate(space(M2bar)[2], space(swgateb2)[2]; Eltype=eltype(M1))
        # codomain(左侧)是M1指标，domain(右侧)是M2指标
        @tensoropt ψ⁺ô□[(l12Dupin, t2Dup, pup1in, r12Dup, Dupin2); (l22Dupin, Dupin, pup2in, r22Dup, b2Dupin)] := 
            envs[x1, y1].corner.lt[lt2t, lt2l1] * envs[x1, y1].transfer.l[lt2l1, l12l2, l12Dup, l12Ddn] *
            envs[x1, y1].transfer.t[lt2t, rt2t, t2Dup, t2Ddn] * envs[x1, y1].corner.rt[rt2t, rt2r1] *
            swgatet1[l12Dup, t2Ddn, l12Dupin, t2Ddnin] * M1bar[l12Ddn, t2Ddnin, pdn1in, r12Ddnin, Ddnin] * 
            swgatet2[pdn1, r12Ddnin2, pdn1in, r12Ddnin] * swgatet3[pup1, r12Ddnin3, pup1in, r12Ddnin2] * 
            swgatet4[Dupin, r12Ddn, Dupin2, r12Ddnin3] * envs[x1, y1].transfer.r[r12Dup, r12Ddn, rt2r1, r12r2] * # toppart
            envs[x2, y2].transfer.l[l12l2, lb2l2, l22Dup, l22Ddn] * envs[x2, y2].corner.lb[lb2l2, lb2b] * 
            swgateb1[Ddnin, l22Dup, Ddninin, l22Dupin3] * M2bar[l22Ddn, Ddninin, pdn2in, r22Ddnin, b2Ddn] * 
            swgateb2[pdn2, l22Dupin3, pdn2in, l22Dupin2] * swgateb3[l22Dupin2, pup2, l22Dupin, pup2in] *
            swgateb4[r22Ddn, b2Dup, r22Ddnin, b2Dupin] * envs[x2, y2].transfer.r[r22Dup, r22Ddn, r12r2, rb2r2] *
            envs[x2, y2].transfer.b[lb2b, b2Dup, b2Ddn, rb2b] * envs[x2, y2].corner.rb[rb2b, rb2r2] * # bottompart
            gate[pdn1, pdn2, pup1, pup2]
        @tensoropt M1grd[l12Dupin, t2Dup, pup1in; r12Dup, Dupin2] :=
            ψ⁺ô□[l12Dupin, t2Dup, pup1in, r12Dup, Dupin2, l22Dupin, Dupin, pup2in, r22Dup, b2Dupin] * 
            M2[l22Dupin, Dupin, pup2in, r22Dup, b2Dupin]
        @tensoropt M2grd[l22Dupin, Dupin, pup2in; r22Dup, b2Dupin] :=
            ψ⁺ô□[l12Dupin, t2Dup, pup1in, r12Dup, Dupin2, l22Dupin, Dupin, pup2in, r22Dup, b2Dupin] * 
            M1[l12Dupin, t2Dup, pup1in, r12Dup, Dupin2]
        @tensoropt nrm = M1grd[l12Dupin, t2Dup, pup1in, r12Dup, Dupin2] * M1[l12Dupin, t2Dup, pup1in, r12Dup, Dupin2]
    else
        error("check input sites")
    end
    ipepsNew = deepcopy(ipeps)
    ipepsNew[x1, y1] = 2.0 * M1grd / nrm
    ipepsNew[x2, y2] = 2.0 * M2grd / nrm
    return ipepsNew.Ms
end

function EnvsTangent_diag(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, Gates::Vector{String}, para::Dict{Symbol,Any}, site1::Vector{Int}, site2::Vector{Int}, get_op::Function; ADflag=true)

end

function EnvsTangent_adj(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, Gates::Vector{String}, para::Dict{Symbol,Any}, site1::Vector{Int}, site2::Vector{Int}, get_op::Function; ADflag=true)

end



# @adjoint Cal_Energy(ipeps::iPEPS, envs::iPEPSenv, get_op::Function, para::Dict{Symbol,Any}) = Cal_Energy(ipeps, envs, get_op, para), Δ -> begin
#     (iPEPS(Δ .* ipeps.Ms, ipeps.Lx, ipeps.Ly),)
# end

# @adjoint iPEPS(Ms, Lx, Ly) = iPEPS(Ms, Lx, Ly), Δ -> begin
#     (iPEPS(Δ .* Ms, Lx::Int, Ly::Int),)
# end

# @adjoint iPEPS(Ms, Lx, Ly) = iPEPS(Ms, Lx, Ly), Δ -> (Δ.Ms, Δ.Lx, Δ.Ly)

# function Zygote.pullback(::Type{iPEPS}, args...)
#     x = iPEPS(args...)
#     function iPEPS_pullback(Δ::iPEPS)
#         # Only propagate gradients for abstractMatrix
#         ∂pepsTensors = Δ.Ms
#         return (∂pepsTensors, nothing, nothing)
#     end
#     return x, iPEPS_pullback
# end


# @adjoint iPEPSenv(Envs, Lx, Ly) = iPEPSenv(Envs, Lx, Ly), Δ -> begin
#     Envs_New = Matrix{_iPEPSenv}(undef, Lx, Ly)
#     for xx in 1:Lx, yy in 1:Ly
#         innerEnv = Envs[xx, yy]  # type::_iPEPSenv
#         corners = innerEnv.corner  # type::_corner
#         transfers = innerEnv.transfer  # type::_transfer
#         cornersNew = _corner(Δ * corners.lt, Δ * corners.lb, Δ * corners.rt, Δ * corners.rb)
#         transfersNew = _transfer(Δ * transfers.l, Δ * transfers.r, Δ * transfers.t, Δ * transfers.b)
#         Envs_New[xx, yy] = _iPEPSenv(cornersNew, transfersNew)
#     end
#     (iPEPSenv(Envs_New, Lx, Ly),)
# end

# @adjoint iPEPSenv(Envs, Lx, Ly) = iPEPSenv(Envs, Lx, Ly), Δ -> (Δ.Envs, Δ.Lx, Δ.Ly)

# Zygote.refresh()