# using MKL
using TensorKit
using Statistics
# import TensorKit.×
# using JLD2

# 测试正方格子 Hubbard 模型, fSU₂. 

include("../iPEPS_Fermionic/iPEPS.jl")
include("../CTMRG_Fermionic/CTMRG.jl")
include("../models/Hubbard.jl")
include("../simple_update_Fermionic/simple_update.jl")
include("../Cal_Obs_Fermionic/Cal_Obs.jl")

function main()
    para = Dict{Symbol,Any}()
    para[:t] = 1.0
    para[:U] = 8.0
    para[:μ] = -4.0
    para[:τlisSU] = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001]
    # para[:τlis] = [1.0]
    para[:maxStep1τ] = 100  # 对每个虚时步长 τ , 最多投影这么多步
    para[:Dk] = 6  # Dkept in the simple udate
    para[:χ] = 100  # env bond dimension
    para[:CTMit] = 20  # CTMRG iteration times
    para[:Etol] = 0.00001  # simple update 能量差小于 para[:Etol]*τ² 这个数就可以继续增大步长
    para[:verbose] = 1
    para[:NNNmethod] = :bond
    para[:pspace] = Rep[U₁×SU₂]((0, 0) => 1, (1, 1 // 2) => 1, (2, 0) => 1)

    pspace = Rep[U₁×SU₂]((0, 0) => 1, (1, 1 // 2) => 1, (2, 0) => 1)
    aspacelr = Rep[U₁×SU₂]((0, 0) => 1, (1, 1 // 2) => 1, (2, 0) => 1)
    aspacetb = Rep[U₁×SU₂]((0, 0) => 1, (1, 1 // 2) => 1, (2, 0) => 1)
    Lx = 2
    Ly = 2
    # 初始化 ΓΛ 形式的 iPEPS, 做 simple update
    ipepsγλ = iPEPSΓΛ(pspace, aspacelr, aspacetb, Lx, Ly; dtype=Float64)
    simple_update!(ipepsγλ, Hubbard_hij, para)
    save(ipepsγλ, para, "/home/tcmp2/JuliaProjects/Hubbard_t$(para[:t])U$(para[:U])mu$(para[:μ])_ipeps_D$(para[:Dk]).jld2")

    # 转换为正常形式, 做 CTMRG 求环境
    ipeps = iPEPS(ipepsγλ)
    @show space(ipeps[1, 1])
    envs = iPEPSenv(ipeps)
    check_qn(ipeps, envs)
    CTMRG!(ipeps, bar(ipeps), envs, para[:χ], para[:CTMit])
    check_qn(ipeps, envs)

    save(ipeps, envs, para, "/home/tcmp2/JuliaProjects/Hubbard_t$(para[:t])U$(para[:U])mu$(para[:μ])_ipepsEnv_D$(para[:Dk])chi$(para[:χ]).jld2")
    GC.gc()
    # 计算观测量
    println("============== Calculating Obs ====================")
    Obs1 = Cal_Obs_2site(ipeps, envs, ["hij", "SS", "NN"], para; site1=[1, 1], site2=[1, 2], get_op=get_op_Hubbard)
    Obs2 = Cal_Obs_2site(ipeps, envs, ["hij", "SS", "NN"], para; site1=[1, 1], site2=[2, 1], get_op=get_op_Hubbard)
    Obs3 = Cal_Obs_2site(ipeps, envs, ["hij", "SS", "NN"], para; site1=[2, 1], site2=[2, 2], get_op=get_op_Hubbard)
    Obs4 = Cal_Obs_2site(ipeps, envs, ["hij", "SS", "NN"], para; site1=[1, 2], site2=[2, 2], get_op=get_op_Hubbard)
    GC.gc()

    Obsdiag1 = Cal_Obs_2site(ipeps, envs, ["hij", "SS", "NN"], para; site1=[1, 1], site2=[2, 2], get_op=get_op_Hubbard)
    Obsdiag2 = Cal_Obs_2site(ipeps, envs, ["hij", "SS", "NN"], para; site1=[1, 2], site2=[2, 1], get_op=get_op_Hubbard)
    @show Obs1
    @show Obs2
    @show Obs3
    @show Obs4
    @show Obsdiag1
    @show Obsdiag2
    Eg = mean(get(Obs1, "hij", NaN) + get(Obs2, "hij", NaN) + get(Obs3, "hij", NaN) + get(Obs4, "hij", NaN)) * 2
    @show Eg

    N11 = Cal_Obs_1site(ipeps, envs, ["N"], para; site=[1, 1], get_op=get_op_Hubbard)
    N12 = Cal_Obs_1site(ipeps, envs, ["N"], para; site=[1, 2], get_op=get_op_Hubbard)
    N21 = Cal_Obs_1site(ipeps, envs, ["N"], para; site=[2, 1], get_op=get_op_Hubbard)
    N22 = Cal_Obs_1site(ipeps, envs, ["N"], para; site=[2, 2], get_op=get_op_Hubbard)
    @show N11, N12, N21, N22
    doping = (get(N11, "N") + get(N12, "N") + get(N21, "N") + get(N22, "N")) / (Lx * Ly)
    @show doping

    return nothing
end

main()