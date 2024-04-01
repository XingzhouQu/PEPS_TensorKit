# using MKL
using TensorKit
import TensorKit.×
# using JLD2

# 测试正方格子 Heisenberg 模型的能量

include("../iPEPS_fSU2/iPEPS.jl")
include("../CTMRG_fSU2/CTMRG.jl")
include("../models/Heisenberg.jl")
include("../simple_update_fSU2/simple_update.jl")
include("../Cal_Obs_fSU2/Cal_Obs.jl")

function main()
    para = Dict{Symbol,Any}()
    para[:J] = 1.0
    para[:τlis] = [0.1, 0.05, 0.02, 0.01, 0.01, 0.005, 0.002, 0.001, 0.001, 0.001]
    para[:Dk] = 10  # Dkept in the simple udate
    para[:pspace] = Rep[SU₂](1 // 2 => 1)

    pspace = Rep[SU₂](1 // 2 => 1)
    aspacelr = Rep[SU₂]((0 => 1), (1 // 2 => 1))
    aspacetb = Rep[SU₂]((0 => 1), (1 // 2 => 1))
    Lx = 2
    Ly = 2
    # 初始化 ΓΛ 形式的 iPEPS, 做 simple update
    ipepsγλ = iPEPSΓΛ(pspace, aspacelr, aspacetb, Lx, Ly; dtype=Float64)
    simple_update!(ipepsγλ, Heisenberg_hij, para)

    # 转换为正常形式, 做 CTMRG 求环境
    ipeps = iPEPS(ipepsγλ)
    @show space(ipeps[1, 1])
    envs = iPEPSenv(ipeps)
    check_qn(ipeps, envs)
    χ = 100
    Nit = 20
    CTMRG!(ipeps, envs, χ, Nit)
    check_qn(ipeps, envs)

    # 计算观测量
    println("============== Calculating Obs ====================")
    E_bond1 = Cal_Obs_2site(ipeps, envs, ["hij"], para; site1=[1, 1], site2=[1, 2], get_op=get_op_Heisenberg)
    E_bond2 = Cal_Obs_2site(ipeps, envs, ["hij"], para; site1=[1, 1], site2=[2, 1], get_op=get_op_Heisenberg)
    E_bond3 = Cal_Obs_2site(ipeps, envs, ["hij"], para, site1=[2, 1], site2=[2, 2], get_op=get_op_Heisenberg)
    E_bond4 = Cal_Obs_2site(ipeps, envs, ["hij"], para, site1=[1, 2], site2=[2, 2], get_op=get_op_Heisenberg)

    @show E_bond1, E_bond2, E_bond3, E_bond4
    @show (get(E_bond1, "hij", NaN) + get(E_bond2, "hij", NaN) + get(E_bond3, "hij", NaN) + get(E_bond4, "hij", NaN)) / 4
end

main()