# using MKL
using TensorKit
using Statistics
# import TensorKit.×
# using JLD2

# 测试正方格子 J1J2 Heisenberg 模型, U1spin. 

include("../iPEPS_fSU2/iPEPS.jl")
include("../CTMRG_fSU2/CTMRG.jl")
include("../models/Heisenberg_U1.jl")
include("../simple_update_fSU2/simple_update.jl")
include("../Cal_Obs_fSU2/Cal_Obs.jl")

function main()
    para = Dict{Symbol,Any}()
    para[:J1] = 1.0
    para[:J2] = 0.3
    para[:τlis] = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001]
    # para[:τlis] = [1.0]
    para[:maxStep1τ] = 50  # 对每个虚时步长 τ , 最多投影这么多步
    para[:Dk] = 6  # Dkept in the simple udate
    para[:χ] = 60  # env bond dimension
    para[:CTMit] = 20  # CTMRG iteration times
    para[:Etol] = 0.001  # simple update 能量差小于 para[:Etol]*τ² 这个数就可以继续增大步长
    para[:verbose] = 1
    para[:NNNmethod] = :bond
    para[:pspace] = Rep[U₁](-1 // 2 => 1, 1 // 2 => 1)

    pspace = Rep[U₁](-1 // 2 => 1, 1 // 2 => 1)
    aspacelr = Rep[U₁](0 => 2, 1 // 2 => 1, -1 // 2 => 1)
    aspacetb = Rep[U₁](0 => 2, 1 // 2 => 1, -1 // 2 => 1)
    Lx = 2
    Ly = 2
    # 初始化 ΓΛ 形式的 iPEPS, 做 simple update
    ipepsγλ = iPEPSΓΛ(pspace, aspacelr, aspacetb, Lx, Ly; dtype=Float64)
    simple_update!(ipepsγλ, J1J2_Heisenberg_hij, para)
    save(ipepsγλ, para, "/home/tcmp2/JuliaProjects/J1$(para[:J1])J2$(para[:J2])_ipeps_D$(para[:Dk]).jld2")

    # 转换为正常形式, 做 CTMRG 求环境
    ipeps = iPEPS(ipepsγλ)
    @show space(ipeps[1, 1])
    envs = iPEPSenv(ipeps)
    check_qn(ipeps, envs)
    χ = para[:χ]
    Nit = para[:CTMit]
    CTMRG!(ipeps, envs, χ, Nit)
    check_qn(ipeps, envs)

    save(ipeps, envs, para, "/home/tcmp2/JuliaProjects/J1$(para[:J1])J2$(para[:J2])_ipepsEnv_D$(para[:Dk])chi$(χ).jld2")
    GC.gc()
    # 计算观测量
    println("============== Calculating Obs ====================")
    Obs1 = Cal_Obs_2site(ipeps, envs, ["hijNN", "SzSz", "SpSm"], para; site1=[1, 1], site2=[1, 2], get_op=get_op_Heisenberg)
    Obs2 = Cal_Obs_2site(ipeps, envs, ["hijNN", "SzSz", "SpSm"], para; site1=[1, 1], site2=[2, 1], get_op=get_op_Heisenberg)
    Obs3 = Cal_Obs_2site(ipeps, envs, ["hijNN", "SzSz", "SpSm"], para; site1=[2, 1], site2=[2, 2], get_op=get_op_Heisenberg)
    Obs4 = Cal_Obs_2site(ipeps, envs, ["hijNN", "SzSz", "SpSm"], para; site1=[1, 2], site2=[2, 2], get_op=get_op_Heisenberg)
    GC.gc()

    Obsdiag1 = Cal_Obs_2site(ipeps, envs, ["hijNNN", "SzSz", "SpSm"], para; site1=[1, 1], site2=[2, 2], get_op=get_op_Heisenberg)
    Obsdiag2 = Cal_Obs_2site(ipeps, envs, ["hijNNN", "SzSz", "SpSm"], para; site1=[1, 2], site2=[2, 1], get_op=get_op_Heisenberg)

    @show Obs1
    @show Obs2
    @show Obs3
    @show Obs4
    @show Obsdiag1
    @show Obsdiag2

    Eg = mean(get(Obs1, "hijNN", NaN) + get(Obs2, "hijNN", NaN) + get(Obs3, "hijNN", NaN) + get(Obs4, "hijNN", NaN)) * 2 +
         mean(get(Obsdiag1, "hijNNN", NaN) + get(Obsdiag2, "hijNNN", NaN)) * 2

    @show Eg
    Sz11 = Cal_Obs_1site(ipeps, envs, ["Sz"], para; site=[1, 1], get_op=get_op_Heisenberg)
    Sz12 = Cal_Obs_1site(ipeps, envs, ["Sz"], para; site=[1, 2], get_op=get_op_Heisenberg)
    Sz21 = Cal_Obs_1site(ipeps, envs, ["Sz"], para; site=[2, 1], get_op=get_op_Heisenberg)
    Sz22 = Cal_Obs_1site(ipeps, envs, ["Sz"], para; site=[2, 2], get_op=get_op_Heisenberg)
    @show Sz11, Sz12, Sz21, Sz22
end

main()