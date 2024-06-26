using MKL
using TensorKit
# import TensorKit.×
# using JLD2

# 测试正方格子 Heisenberg 模型, U1spin. 

include("../iPEPS_Bosonic/iPEPS.jl")
include("../CTMRG_Bosonic/CTMRG.jl")
include("../models/Heisenberg_U1.jl")
include("../simple_update_Bosonic/simple_update.jl")
include("../Cal_Obs_Bosonic/Cal_Obs.jl")

function main()
    para = Dict{Symbol,Any}()
    para[:J] = 1.0
    para[:τlisSU] = [1.0, 0.5, 0.1, 0.05, 0.01, 0.001, 0.0001]
    para[:maxStep1τ] = 50  # 对每个虚时步长 τ , 最多投影这么多步
    para[:Dk] = 6  # Dkept in the simple udate
    para[:χ] = 100  # env bond dimension
    para[:CTMit] = 20  # CTMRG iteration times
    para[:Etol] = 0.0001  # simple update 能量差小于 para[:Etol]*τ² 这个数就可以继续增大步长
    para[:verbose] = 1
    para[:NNNmethod] = :bond
    para[:pspace] = Rep[U₁](-1 // 2 => 1, 1 // 2 => 1)

    pspace = Rep[U₁](-1 // 2 => 1, 1 // 2 => 1)
    aspacelr = Rep[U₁](0 => 1, 1 // 2 => 1, -1 // 2 => 1)
    aspacetb = Rep[U₁](0 => 1, 1 // 2 => 1, -1 // 2 => 1)
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
    E_bond1 = Cal_Obs_2site(ipeps, envs, ["hij", "SzSz", "SpSm"], para; site1=[1, 1], site2=[1, 2], get_op=get_op_Heisenberg)
    E_bond2 = Cal_Obs_2site(ipeps, envs, ["hij", "SzSz", "SpSm"], para; site1=[1, 1], site2=[2, 1], get_op=get_op_Heisenberg)
    E_bond3 = Cal_Obs_2site(ipeps, envs, ["hij", "SzSz", "SpSm"], para, site1=[2, 1], site2=[2, 2], get_op=get_op_Heisenberg)
    E_bond4 = Cal_Obs_2site(ipeps, envs, ["hij", "SzSz", "SpSm"], para, site1=[1, 2], site2=[2, 2], get_op=get_op_Heisenberg)

    @show E_bond1, E_bond2, E_bond3, E_bond4
    @show (get(E_bond1, "hij", NaN) + get(E_bond2, "hij", NaN) + get(E_bond3, "hij", NaN) + get(E_bond4, "hij", NaN)) / 2

    Sz11 = Cal_Obs_1site(ipeps, envs, ["Sz"], para; site=[1, 1], get_op=get_op_Heisenberg)
    Sz12 = Cal_Obs_1site(ipeps, envs, ["Sz"], para; site=[1, 2], get_op=get_op_Heisenberg)
    Sz21 = Cal_Obs_1site(ipeps, envs, ["Sz"], para; site=[2, 1], get_op=get_op_Heisenberg)
    Sz22 = Cal_Obs_1site(ipeps, envs, ["Sz"], para; site=[2, 2], get_op=get_op_Heisenberg)
    @show Sz11, Sz12, Sz21, Sz22
end

main()