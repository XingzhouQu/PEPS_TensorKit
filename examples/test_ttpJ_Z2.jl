# using MKL
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

# 测试正方格子 t-t'-J 模型加 Zeeman 场, Z₂charge. 

include("../iPEPS_Fermionic/iPEPS.jl")
include("../CTMRG_Fermionic/CTMRG.jl")
include("../models/tJ_Z2.jl")
include("../simple_update_Fermionic/simple_update.jl")
include("../fast_full_update_Fermionic/fast_full_update.jl")
include("../Cal_Obs_Fermionic/Cal_Obs.jl")

function main()
    para = Dict{Symbol,Any}()
    para[:t] = 3.0
    para[:tp] = 0.51
    para[:J] = 1.0
    para[:Jp] = 0.0289
    para[:h] = 0.6
    para[:μ] = 5.0
    para[:τlisSU] = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001]
    para[:τlisFFU] = [0.01, 0.005, 0.001, 0.0001]
    para[:minStep1τ] = 50   # 对每个虚时步长 τ , 最少投影这么多步
    para[:maxStep1τ] = 200  # 对每个虚时步长 τ , 最多投影这么多步
    para[:maxiterFFU] = 60
    para[:tolFFU] = 1e-10  # FFU 中损失函数的 Tolerence
    para[:Dk] = 6  # Dkept in the simple udate
    para[:χ] = 140  # env bond dimension
    para[:CTMit] = 20  # CTMRG iteration times
    para[:CTMparallel] = false  # use parallel CTMRG or not. Use with MKL.
    para[:Etol] = 1e-6  # simple update 能量差小于 para[:Etol]*τ² 这个数就可以继续增大步长. 1e-5对小size
    para[:verbose] = 1
    para[:NNNmethod] = :bond
    para[:pspace] = Rep[ℤ₂](0 => 1, 1 => 2)

    pspace = Rep[ℤ₂](0 => 1, 1 => 2)
    aspacelr = Rep[ℤ₂](0 => 1, 1 => 2)
    aspacetb = Rep[ℤ₂](0 => 1, 1 => 2)
    Lx = 2
    Ly = 2
    # # 决定初态每条腿的量子数
    # aspacel = Matrix{GradedSpace}(undef, Lx, Ly)
    # aspacet = Matrix{GradedSpace}(undef, Lx, Ly)
    # aspacer = Matrix{GradedSpace}(undef, Lx, Ly)
    # aspaceb = Matrix{GradedSpace}(undef, Lx, Ly)
    # # TODO: 合理选择每条腿的量子数，达到固定的掺杂
    # ipepsγλ = iPEPSΓΛ(pspace, aspacel, aspacet, aspacer, aspaceb, Lx, Ly; dtype=Float64)

    # simple update
    ipepsγλ = iPEPSΓΛ(pspace, aspacelr, aspacetb, Lx, Ly; dtype=Float64)
    simple_update!(ipepsγλ, tJ_hij, para)
    # save(ipepsγλ, para, "/home/tcmp2/JuliaProjects/tJZ2_Lx$(Lx)Ly$(Ly)_t$(para[:t])t'$(para[:tp])J$(para[:J])J'$(para[:Jp])h$(para[:h])mu$(para[:μ])_ipeps_D$(para[:Dk]).jld2")

    # 转换为正常形式, 做 fast full update
    ipeps = iPEPS(ipepsγλ)
    @show space(ipeps[1, 1])
    ipepsbar = bar(ipeps)
    envs = iPEPSenv(ipeps)
    # check_qn(ipeps, envs)
    # CTMRG!(ipeps, ipepsbar, envs, para[:χ], 2)
    # fast_full_update!(ipeps, envs, Hubbard_hij, para)
    # check_qn(ipeps, envs)

    # 最后再做CTMRG
    CTMRG!(ipeps, ipepsbar, envs, para[:χ], para[:CTMit]; parallel=para[:CTMparallel])
    # save(ipeps, envs, para, "/home/tcmp2/JuliaProjects/tJZ2_Lx$(Lx)Ly$(Ly)_SU_t$(para[:t])t'$(para[:tp])J$(para[:J])J'$(para[:Jp])h$(para[:h])mu$(para[:μ])_ipepsEnv_D$(para[:Dk])chi$(para[:χ]).jld2")
    GC.gc()
    # 计算观测量
    println("============== Calculating Obs ====================")
    site1Obs = ["N", "Sz"]             # 计算这些单点观测量
    site2Obs = ["hijNN", "SpSm", "SzSz", "NN"]   # 计算这些两点观测量
    site2Obsdiag = ["hijNNN", "SpSm", "SzSz", "NN"]
    # sites = [[x, y] for x in 1:Lx, y in 1:Ly]
    # @floop 
    @floop for ind in CartesianIndices((Lx, Ly))
        (xx, yy) = Tuple(ind)
        Obs1si = Cal_Obs_1site(ipeps, ipepsbar, envs, site1Obs, para; site=[xx, yy], get_op=get_op_tJ)
        @show (xx, yy, Obs1si)
        @reduce filling += get(Obs1si, "N", NaN)
        @reduce magnetization += abs(get(Obs1si, "Sz", NaN))

        Obs2si_h = Cal_Obs_2site(ipeps, ipepsbar, envs, site2Obs, para; site1=[xx, yy], site2=[xx + 1, yy], get_op=get_op_tJ)
        @show (xx, yy, Obs2si_h)
        Obs2si_v = Cal_Obs_2site(ipeps, ipepsbar, envs, site2Obs, para; site1=[xx, yy], site2=[xx, yy + 1], get_op=get_op_tJ)
        @show (xx, yy, Obs2si_v)
        @reduce Eg += (get(Obs2si_h, "hijNN", NaN) + get(Obs2si_v, "hijNN", NaN))

        Obs2sidiag_lurd = Cal_Obs_2site(ipeps, ipepsbar, envs, site2Obsdiag, para; site1=[xx, yy], site2=[xx + 1, yy + 1], get_op=get_op_tJ)
        Obs2sidiag_ruld = Cal_Obs_2site(ipeps, ipepsbar, envs, site2Obsdiag, para; site1=[xx, yy], site2=[xx - 1, yy + 1], get_op=get_op_tJ)
        @reduce Eg += (get(Obs2sidiag_lurd, "hijNNN", NaN) + get(Obs2sidiag_ruld, "hijNNN", NaN))
        GC.gc()
    end
    GC.gc()
    filling = filling / (Lx * Ly)
    @show filling
    magnetization = magnetization / (Lx * Ly)
    @show magnetization
    Eg = Eg / (Lx * Ly) + filling * para[:μ]
    @show Eg

    return nothing
end

main()