using MKL
using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(8)
using TensorOperations, TensorKit
using Statistics
import TensorKit.×
using JLD2
using Strided, FLoops
Strided.enable_threads()
@show Threads.nthreadpools()
@show Threads.nthreads()

# 测试正方格子 Hubbard 模型, U₁charge × SU₂spin. 

include("../iPEPS_Fermionic/iPEPS.jl")
include("../CTMRG_Fermionic/CTMRG.jl")
include("../models/Hubbard_Z2SU2.jl")
include("../simple_update_Fermionic/simple_update.jl")
include("../Cal_Obs_Fermionic/Cal_Obs.jl")
include("../AD/ADutils.jl")

function main()
    para = Dict{Symbol,Any}()
    para[:t] = 1.0
    para[:U] = 8.0
    para[:μ] = 4.0
    para[:τlisSU] = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001]
    para[:τlisFFU] = [0.01, 0.005, 0.001, 0.0001]
    para[:minStep1τ] = 10  # 对每个虚时步长 τ , 最少投影这么多步
    para[:maxStep1τ] = 2000  # 对每个虚时步长 τ , 最多投影这么多步
    para[:maxiterFFU] = 60
    para[:tolFFU] = 1e-10  # FFU 中损失函数的 Tolerence
    para[:Dk] = 8  # Dkept in the simple udate
    para[:χ] = 150  # env bond dimension
    para[:CTMit] = 20  # CTMRG iteration times
    para[:CTMparallel] = true  # use parallel CTMRG or not
    para[:Etol] = 1e-6  # simple update 能量差小于 para[:Etol]*τ² 这个数就可以继续增大步长. 1e-5对小size
    para[:verbose] = 1
    para[:TrotterOrder] = 2 # 用几阶Trotter分解,设为1或2
    para[:pspace] = Rep[ℤ₂×SU₂]((0, 0) => 2, (1, 1 // 2) => 1)

    pspace = Rep[ℤ₂×SU₂]((0, 0) => 2, (1, 1 // 2) => 1)
    aspacelr = Rep[ℤ₂×SU₂]((0, 0) => 2, (1, 1 // 2) => 1)
    aspacetb = Rep[ℤ₂×SU₂]((0, 0) => 2, (1, 1 // 2) => 1)
    Lx = 2
    Ly = 2

    # initialize
    ipepsγλ = iPEPSΓΛ(pspace, aspacelr, aspacetb, Lx, Ly; dtype=Float64)
    # save(ipepsγλ, para, "/home/tcmp2/JuliaProjects/HubbardZ2SU2_Lx$(Lx)Ly$(Ly)_t$(para[:t])U$(para[:U])mu$(para[:μ])_ipeps_D$(para[:Dk]).jld2")
    ipeps = iPEPS(ipepsγλ)

    for ii in 1:para[:ADit]
        envs = iPEPSenv(ipeps)
        CTMRG!(ipeps, ipepsbar, envs, para[:χ], para[:CTMit]; parallel=para[:CTMparallel])
        energy, ∂H_∂A = loss_Energy(ipeps, envs, Cal_Obs_2site, CTMRG!, para)
        ipeps, energy, grd, numfg, normgradhistory = optimize(loss_Energy, ipeps, LBFGS(20))
        println("AD iteration $ii / $(para[:ADit]), energy $energy")
        println()
    end
    # check_qn(ipeps, envs)
    # CTMRG!(ipeps, ipepsbar, envs, para[:χ], 2)
    # fast_full_update!(ipeps, envs, Hubbard_hij, para)
    # check_qn(ipeps, envs)

    # 最后再做CTMRG
    CTMRG!(ipeps, ipepsbar, envs, para[:χ], para[:CTMit]; parallel=para[:CTMparallel])
    # save(ipeps, envs, para, "/home/tcmp2/JuliaProjects/HubbardZ2SU2_Lx$(Lx)Ly$(Ly)_SU_t$(para[:t])U$(para[:U])mu$(para[:μ])_ipepsEnv_D$(para[:Dk])chi$(para[:χ]).jld2")
    GC.gc()
    # 计算观测量
    println("============== Calculating Obs ====================")
    site1Obs = ["N"]                                # 计算这些单点观测量
    site2Obs = ["hij", "SS", "NN", "Δₛ", "Δₛdag"]   # 计算这些两点观测量
    # sites = [[x, y] for x in 1:Lx, y in 1:Ly]
    # @floop 
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