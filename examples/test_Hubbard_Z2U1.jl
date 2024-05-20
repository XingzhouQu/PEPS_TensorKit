using MKL
using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(4)
using TensorOperations, TensorKit
using Statistics
import TensorKit.×
using JLD2
using Strided, FLoops
Strided.enable_threads()
@show Threads.nthreadpools()
@show Threads.nthreads()

# 测试正方格子 Hubbard 模型, Z₂charge × U₁spin. 

include("../iPEPS_Fermionic/iPEPS.jl")
include("../CTMRG_Fermionic/CTMRG.jl")
include("../models/Hubbard_Z2U1.jl")
include("../simple_update_Fermionic/simple_update.jl")
include("../Cal_Obs_Fermionic/Cal_Obs.jl")

function main()
    para = Dict{Symbol,Any}()
    para[:t] = 1.0
    para[:U] = 8.0
    para[:μ] = 4.0
    para[:τlisSU] = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001]
    # para[:τlis] = [1.0]
    para[:maxStep1τ] = 200  # 对每个虚时步长 τ , 最多投影这么多步
    para[:Dk] = 6  # Dkept in the simple udate
    para[:χ] = 100  # env bond dimension
    para[:CTMit] = 20  # CTMRG iteration times
    para[:Etol] = 0.000001  # simple update 能量差小于 para[:Etol]*τ² 这个数就可以继续增大步长
    para[:verbose] = 1
    para[:NNNmethod] = :bond
    para[:CTMparallel] = false  # contract CTMRG env in parallel or not. Better with MKL??
    para[:pspace] = Rep[ℤ₂×U₁]((0, 0) => 2, (1, 1 // 2) => 1, (1, -1 // 2) => 1)

    pspace = Rep[ℤ₂×U₁]((0, 0) => 2, (1, 1 // 2) => 1, (1, -1 // 2) => 1)
    aspacelr = Rep[ℤ₂×U₁]((0, 0) => 2, (1, 1 // 2) => 1, (1, -1 // 2) => 1)
    aspacetb = Rep[ℤ₂×U₁]((0, 0) => 2, (1, 1 // 2) => 1, (1, -1 // 2) => 1)
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
    # save(ipepsγλ, para, "/home/tcmp2/JuliaProjects/HubbardZ2U1_t$(para[:t])U$(para[:U])mu$(para[:μ])_ipeps_D$(para[:Dk]).jld2")

    # 转换为正常形式, 做 CTMRG 求环境
    ipeps = iPEPS(ipepsγλ)
    @show space(ipeps[1, 1])
    ipepsbar = bar(ipeps)
    envs = iPEPSenv(ipeps)
    check_qn(ipeps, envs)
    CTMRG!(ipeps, ipepsbar, envs, para[:χ], para[:CTMit]; parallel=para[:CTMparallel])
    check_qn(ipeps, envs)

    # save(ipeps, envs, para, "/home/tcmp2/JuliaProjects/HubbardZ2U1_t$(para[:t])U$(para[:U])mu$(para[:μ])_ipepsEnv_D$(para[:Dk])chi$(para[:χ]).jld2")
    GC.gc()
    # 计算观测量
    println("============== Calculating Obs ====================")
    site1Obs = ["N", "Sz"]                                # 计算这些单点观测量
    site2Obs = ["hij", "SpSm", "NN", "SzSz"]   # 计算这些两点观测量
    # sites = [[x, y] for x in 1:Lx, y in 1:Ly]
    @floop for ind in CartesianIndices((Lx, Ly))
        (xx, yy) = Tuple(ind)
        Obs1si = Cal_Obs_1site(ipeps, ipepsbar, envs, site1Obs, para; site=[xx, yy], get_op=get_op_Hubbard)
        @show Obs1si
        @reduce doping += get(Obs1si, "N", NaN)

        GC.gc()
        Obs2si_h = Cal_Obs_2site(ipeps, ipepsbar, envs, site2Obs, para; site1=[xx, yy], site2=[xx + 1, yy], get_op=get_op_Hubbard)
        @show Obs2si_h
        Obs2si_v = Cal_Obs_2site(ipeps, ipepsbar, envs, site2Obs, para; site1=[xx, yy], site2=[xx, yy + 1], get_op=get_op_Hubbard)
        @show Obs2si_v
        @reduce Eg += (get(Obs2si_h, "hij", NaN) + get(Obs2si_v, "hij", NaN))
        GC.gc()

        # Obs2sidiag = Cal_Obs_2site(ipeps, ipepsbar, envs, site2Obs, para; site1=[xx, yy], site2=[xx+1, yy+1], get_op=get_op_Hubbard)
        # @show Obs2sidiag
        # Obs2sidiag = Cal_Obs_2site(ipeps, ipepsbar, envs, site2Obs, para; site1=[xx, yy], site2=[xx - 1, yy + 1], get_op=get_op_Hubbard)
    end
    GC.gc()
    doping = doping / (Lx * Ly)
    @show doping
    Eg = Eg / (Lx * Ly) + doping * para[:μ]
    @show Eg

    return nothing
end

main()