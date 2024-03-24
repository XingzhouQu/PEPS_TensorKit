using MKL
using TensorKit
import TensorKit.×
using JLD2

include("../iPEPS_fSU2/iPEPS.jl")
include("../CTMRG_fSU2/CTMRG.jl")
include("../simple_update_fSU2/simple_update.jl")

function main()
    para = Dict{Symbol,Any}()
    para[:t] = 1.0
    para[:U] = 8
    para[:μ] = 0.4
    para[:τlis] = [0.1, 0.01, 1e-3, 1e-4, 1e-5]
    para[:Dk] = 8  # Dkept in the simple udate
    para[:pspace] = GradedSpace{fSU₂}((0 => 2), (1 // 2 => 1))

    pspace = GradedSpace{fSU₂}((0 => 2), (1 // 2 => 1))
    aspacelr = GradedSpace{fSU₂}((0 => 1), (1 // 2 => 2), (1 => 1))
    aspacetb = GradedSpace{fSU₂}((0 => 1), (1 // 2 => 2), (1 => 1))
    Lx = 2
    Ly = 2
    # 初始化 ΓΛ 形式的 iPEPS, 做 simple update
    ipepsγλ = iPEPSΓΛ(pspace, aspacelr, aspacetb, Lx, Ly; dtype=Float64)
    simple_update!(ipepsγλ, para)

    # 转换为正常形式, 做 CTMRG 求环境
    ipeps = iPEPS(ipepsγλ)
    for x in 1:Lx, y in 1:Ly
        @assert space(ipeps[x, y])[1] == space(ipeps[x, y])[4]' "ipeps[$x, $y] left and right space mismatch"
        @assert space(ipeps[x, y])[2] == space(ipeps[x, y])[5]' "ipeps[$x, $y] top and bottom space mismatch"
    end
    envs = iPEPSenv(ipeps)
    χ = 10
    Nit = 2
    CTMRG!(ipeps, envs, χ, Nit)

    # 计算观测量
    E_bond1 = Cal_Obs_2site(ipeps, envs, ["hij"], para; site1=[1, 1], site2=[1, 2])
    E_bond2 = Cal_Obs_2site(ipeps, envs, ["hij"], para; site1=[1, 1], site2=[2, 1])
    E_bond3 = Cal_Obs_2site(ipeps, envs, ["hij"], para; site1=[2, 1], site2=[2, 2])
    E_bond4 = Cal_Obs_2site(ipeps, envs, ["hij"], para; site1=[1, 2], site2=[2, 2])

    @show E_bond1, E_bond2, E_bond3, E_bond4
    @show (E_bond1 + E_bond2 + E_bond3 + E_bond4) / 4
end

main()