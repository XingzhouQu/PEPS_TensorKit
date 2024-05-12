# using MKL
using TensorKit
using Statistics
import TensorKit.×
# using JLD2

# 测试正方格子 Hubbard 模型, U₁charge × SU₂spin. 

include("../iPEPS_Fermionic/iPEPS.jl")
include("../CTMRG_Fermionic/CTMRG.jl")
include("../models/Hubbard_Z2SU2.jl")
include("../simple_update_Fermionic/simple_update.jl")
include("../Cal_Obs_Fermionic/Cal_Obs.jl")

function main()
    para = Dict{Symbol,Any}()
    para[:t] = 1.0
    para[:U] = 8.0
    para[:μ] = 5.0
    para[:τlisSU] = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001]
    # para[:τlis] = [1.0]
    para[:maxStep1τ] = 100  # 对每个虚时步长 τ , 最多投影这么多步
    para[:Dk] = 8  # Dkept in the simple udate
    para[:χ] = 150  # env bond dimension
    para[:CTMit] = 30  # CTMRG iteration times
    para[:Etol] = 0.00001  # simple update 能量差小于 para[:Etol]*τ² 这个数就可以继续增大步长
    para[:verbose] = 1
    para[:NNNmethod] = :bond
    para[:pspace] = Rep[ℤ₂×SU₂]((0, 0) => 2, (1, 1 // 2) => 1)

    pspace = Rep[ℤ₂×SU₂]((0, 0) => 2, (1, 1 // 2) => 1)
    aspacelr = Rep[ℤ₂×SU₂]((0, 0) => 2, (1, 1 // 2) => 1)
    aspacetb = Rep[ℤ₂×SU₂]((0, 0) => 2, (1, 1 // 2) => 1)
    Lx = 2
    Ly = 2
    # # 决定初态每条腿的量子数
    # aspacel = Matrix{GradedSpace}(undef, Lx, Ly)
    # aspacet = Matrix{GradedSpace}(undef, Lx, Ly)
    # aspacer = Matrix{GradedSpace}(undef, Lx, Ly)
    # aspaceb = Matrix{GradedSpace}(undef, Lx, Ly)
    # # TODO: 合理选择每条腿的量子数，达到固定的掺杂
    # ipepsγλ = iPEPSΓΛ(pspace, aspacel, aspacet, aspacer, aspaceb, Lx, Ly; dtype=Float64)

    ipepsγλ = iPEPSΓΛ(pspace, aspacelr, aspacetb, Lx, Ly; dtype=Float64)
    simple_update!(ipepsγλ, Hubbard_hij, para)
    save(ipepsγλ, para, "/home/tcmp2/JuliaProjects/Hubbard_t$(para[:t])U$(para[:U])mu$(para[:μ])_ipeps_D$(para[:Dk]).jld2")

    # 转换为正常形式, 做 CTMRG 求环境
    ipeps = iPEPS(ipepsγλ)
    @show space(ipeps[1, 1])
    ipepsbar = bar(ipeps)
    envs = iPEPSenv(ipeps)
    check_qn(ipeps, envs)
    CTMRG!(ipeps, ipepsbar, envs, para[:χ], para[:CTMit])
    check_qn(ipeps, envs)

    save(ipeps, envs, para, "/home/tcmp2/JuliaProjects/Hubbard_t$(para[:t])U$(para[:U])mu$(para[:μ])_ipepsEnv_D$(para[:Dk])chi$(para[:χ]).jld2")
    GC.gc()
    # 计算观测量
    println("============== Calculating Obs ====================")
    site1Obs = ["N"]                                # 计算这些单点观测量
    site2Obs = ["hij", "SS", "NN", "Δₛ", "Δₛdag"]   # 计算这些两点观测量
    # sites = [[x, y] for x in 1:Lx, y in 1:Ly]
    Eg = 0.0
    doping = 0.0
    for xx in 1:Lx, yy in 1:Ly
        Obs1si = Cal_Obs_1site(ipeps, ipepsbar, envs, site1Obs, para; site=[xx, yy], get_op=get_op_Hubbard)
        @show Obs1si
        doping += get(Obs1si, "N", NaN)

        Obs2si = Cal_Obs_2site(ipeps, ipepsbar, envs, site2Obs, para; site1=[xx, yy], site2=[xx + 1, yy], get_op=get_op_Hubbard)
        @show Obs2si
        Eg += get(Obs2si, "hij", NaN)
        Obs2si = Cal_Obs_2site(ipeps, ipepsbar, envs, site2Obs, para; site1=[xx, yy], site2=[xx, yy + 1], get_op=get_op_Hubbard)
        @show Obs2si
        Eg += get(Obs2si, "hij", NaN)

        # Obs2sidiag = Cal_Obs_2site(ipeps, ipepsbar, envs, site2Obs, para; site1=[xx, yy], site2=[xx+1, yy+1], get_op=get_op_Hubbard)
        # @show Obs2sidiag
        # Obs2sidiag = Cal_Obs_2site(ipeps, ipepsbar, envs, site2Obs, para; site1=[xx, yy], site2=[xx - 1, yy + 1], get_op=get_op_Hubbard)
    end
    GC.gc()
    Eg = Eg / (Lx * Ly)
    @show Eg
    doping = doping / (Lx * Ly)
    @show doping

    return nothing
end

main()