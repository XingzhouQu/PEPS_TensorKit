using ChainRulesCore

"""
Custom Gradient: Use Zygote.@adjoint to specify how to compute the gradient of f(x) = x^3 manually.

Zygote.@adjoint f(x) = f(x), Δ -> (3x^2 * Δ,)

The first part, f(x), specifies the value of the function.

The second part, Δ -> (3x^2 * Δ,), specifies how to compute the gradient. 
Δ represents the upstream gradient (the gradient of the loss with respect to the output of f).
"""

# 这里要确保梯度流 Δ 都是 TensorMap 类型？
"""
Trying to define basic adjoints for local obs functions here

    since ∂Tr(XᵀBX)/∂X = BX + BᵀX, for an Hermitian operator ô = ô⁺ and real wave function ψ, 
    we have ∂Tr(ψ⁺ôψ)/∂ψ = 2ôψ
"""
function ChainRulesCore.rrule(::typeof(_2siteObs_diagSite), args...)
    # args = ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, Gates::Vector{String}, para::Dict{Symbol,Any}, site1::Vector{Int}, site2::Vector{Int}, get_op::Function
    y = _2siteObs_diagSite(args...)
    function diagObsPullback(ȳ)
        f̄ = NoTangent()  # The function Cal_Obs has no fields (i.e. it is not a closure) and can not be perturbed. Therefore its tangent (f̄) is a NoTangent.
        # The struct ipeps::iPEPS gets a Tangent{iPEPS} structural tangent, which stores the tangents of fields of ipeps.
        #     The tangent of the field Ms is specified,
        #     The tangent of the field Lx and Ly are NoTangent(), because they can not be perturbed, either.
        ipeps̄ = Tangent{iPEPS}(; Ms=MsTangent_diag(args...), Lx=NoTangent(), Ly=NoTangent())
        envs̄ = Tangent{iPEPSenv}(; Envs=EnvsTangent_diag(args...), Lx=NoTangent(), Ly=NoTangent())
        
        return f̄, ipeps̄, envs̄
    end
    return y, diagObsPullback
end


function ChainRulesCore.rrule(::typeof(_2siteObs_adjSite), args...)
    # args = ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, Gates::Vector{String}, para::Dict{Symbol,Any}, site1::Vector{Int}, site2::Vector{Int}, get_op::Function
    y = _2siteObs_adjSite(args...)
    function diagObsPullback(ȳ)
        f̄ = NoTangent()  # The function Cal_Obs has no fields (i.e. it is not a closure) and can not be perturbed. Therefore its tangent (f̄) is a NoTangent.
        # The struct ipeps::iPEPS gets a Tangent{iPEPS} structural tangent, which stores the tangents of fields of ipeps.
        #     The tangent of the field Ms is specified,
        #     The tangent of the field Lx and Ly are NoTangent(), because they can not be perturbed, either.
        ipeps̄ = Tangent{iPEPS}(; Ms=MsTangent_adj(args...), Lx=NoTangent(), Ly=NoTangent())
        envs̄ = Tangent{iPEPSenv}(; Envs=EnvsTangent_adj(args...), Lx=NoTangent(), Ly=NoTangent())
        
        return f̄, ipeps̄, envs̄
    end
    return y, diagObsPullback
end

function MsTangent_diag(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, Gates::Vector{String}, para::Dict{Symbol,Any}, site1::Vector{Int}, site2::Vector{Int}, get_op::Function; ADflag=true)

end

function MsTangent_adj(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, Gates::Vector{String}, para::Dict{Symbol,Any}, site1::Vector{Int}, site2::Vector{Int}, get_op::Function; ADflag=true)

end

function EnvsTangent_diag(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, Gates::Vector{String}, para::Dict{Symbol,Any}, site1::Vector{Int}, site2::Vector{Int}, get_op::Function; ADflag=true)

end

function EnvsTangent_adj(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, Gates::Vector{String}, para::Dict{Symbol,Any}, site1::Vector{Int}, site2::Vector{Int}, get_op::Function; ADflag=true)

end

@adjoint _2siteObs_adjSite(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, Gates::Vector{String}, para::Dict{Symbol,Any},
site1::Vector{Int}, site2::Vector{Int}, get_op::Function; ADflag=true) = _2siteObs_adjSite(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, Gates::Vector{String}, para::Dict{Symbol,Any},
site1::Vector{Int}, site2::Vector{Int}, get_op::Function; ADflag=true), Δ -> begin
    ADflag ? ipepsbar = ignore_derivatives(ipepsbar) : nothing
    Lx = ipeps.Lx
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
        @tensor nrm = ψ□ψ[p1, p2, p1, p2]
    else
        error("check input sites")
    end
    gate = get_op(Gates[1], para)
    @tensor val = ψ□ψ[pup1, pup2, pdn1, pdn2] * gate[pdn1, pdn2, pup1, pup2]
    
    Δ * ψH□
end


# @adjoint Cal_Energy(ipeps::iPEPS, envs::iPEPSenv, get_op::Function, para::Dict{Symbol,Any}) = Cal_Energy(ipeps, envs, get_op, para), Δ -> begin
#     (iPEPS(Δ .* ipeps.Ms, ipeps.Lx, ipeps.Ly),)
# end

# @adjoint iPEPS(Ms, Lx, Ly) = iPEPS(Ms, Lx, Ly), Δ -> begin
#     (iPEPS(Δ .* Ms, Lx::Int, Ly::Int),)
# end

# @adjoint iPEPS(Ms, Lx, Ly) = iPEPS(Ms, Lx, Ly), Δ -> (Δ.Ms, Δ.Lx, Δ.Ly)

function Zygote.pullback(::Type{iPEPS}, args...)
    x = iPEPS(args...)
    function iPEPS_pullback(Δ::iPEPS)
        # Only propagate gradients for abstractMatrix
        ∂pepsTensors = Δ.Ms
        return (∂pepsTensors, nothing, nothing)
    end
    return x, iPEPS_pullback
end


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

Zygote.refresh()