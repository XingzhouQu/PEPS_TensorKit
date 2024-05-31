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

# 测试正方格子 J1J2 Heisenberg 模型, U1spin. 

include("../iPEPS_Bosonic/iPEPS.jl")
include("../CTMRG_Bosonic/CTMRG.jl")
include("../models/Heisenberg_U1.jl")
include("../simple_update_Bosonic/simple_update.jl")
include("../Cal_Obs_Bosonic/Cal_Obs.jl")

function main()
    para = Dict{Symbol,Any}()
    para[:J1] = 1.0
    para[:J2] = 0.8
    para[:τlisSU] = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001]
    para[:minStep1τ] = 50   # 对每个虚时步长 τ , 最少投影这么多步
    para[:maxStep1τ] = 150  # 对每个虚时步长 τ , 最多投影这么多步
    para[:Dk] = 8  # Dkept in the simple udate
    para[:χ] = 100  # env bond dimension
    para[:CTMit] = 20  # CTMRG iteration times
    para[:Etol] = 0.00001  # simple update 能量差小于 para[:Etol]*τ² 这个数就可以继续增大步长
    para[:verbose] = 1
    para[:NNNmethod] = :bond
    para[:pspace] = Rep[U₁](-1 // 2 => 1, 1 // 2 => 1)

    pspace = Rep[U₁](-1 // 2 => 1, 1 // 2 => 1)
    aspacelr = Rep[U₁](0 => 2, 1 // 2 => 1, -1 // 2 => 1)
    aspacetb = Rep[U₁](0 => 2, 1 // 2 => 1, -1 // 2 => 1)
    Lx = 4
    Ly = 4
    # 初始化 ΓΛ 形式的 iPEPS, 做 simple update
    ipepsγλ = iPEPSΓΛ(pspace, aspacelr, aspacetb, Lx, Ly; dtype=Float64)
    simple_update!(ipepsγλ, J1J2_Heisenberg_hij, para)
    save(ipepsγλ, para, "/home/tcmp2/JuliaProjects/HeisenbergU1_Lx$(Lx)Ly$(Ly)_SU_J1$(para[:J1])J2$(para[:J2])_ipeps_D$(para[:Dk]).jld2")

    # 转换为正常形式, 做 CTMRG 求环境
    ipeps = iPEPS(ipepsγλ)
    @show space(ipeps[1, 1])
    envs = iPEPSenv(ipeps)
    check_qn(ipeps, envs)
    χ = para[:χ]
    Nit = para[:CTMit]
    CTMRG!(ipeps, envs, χ, Nit)
    check_qn(ipeps, envs)

    save(ipeps, envs, para, "/home/tcmp2/JuliaProjects/HeisenbergU1_Lx$(Lx)Ly$(Ly)_SU_J1$(para[:J1])J2$(para[:J2])_ipepsEnv_D$(para[:Dk])chi$(χ).jld2")
    GC.gc()

    # 计算观测量
    println("============== Calculating Obs ====================")
    site1Obs = ["Sz"]             # 计算这些单点观测量
    site2Obs = ["hijNN", "SzSz", "SpSm"]   # 计算这些两点观测量
    site2Obsdiag = ["hijNNN", "SzSz", "SpSm"]
    @floop for ind in CartesianIndices((Lx, Ly))
        (xx, yy) = Tuple(ind)
        Obs1si = Cal_Obs_1site(ipeps, envs, site1Obs, para; site=[xx, yy], get_op=get_op_Heisenberg)
        @show (xx, yy, Obs1si)
        @reduce magnetization += abs(get(Obs1si, "Sz", NaN))

        Obs2si_h = Cal_Obs_2site(ipeps, envs, site2Obs, para; site1=[xx, yy], site2=[xx + 1, yy], get_op=get_op_Heisenberg)
        @show (xx, yy, Obs2si_h)
        Obs2si_v = Cal_Obs_2site(ipeps, envs, site2Obs, para; site1=[xx, yy], site2=[xx, yy + 1], get_op=get_op_Heisenberg)
        @show (xx, yy, Obs2si_v)
        @reduce Eg += (get(Obs2si_h, "hijNN", NaN) + get(Obs2si_v, "hijNN", NaN))

        Obs2sidiag_lurd = Cal_Obs_2site(ipeps, envs, site2Obsdiag, para; site1=[xx, yy], site2=[xx + 1, yy + 1], get_op=get_op_Heisenberg)
        Obs2sidiag_ruld = Cal_Obs_2site(ipeps, envs, site2Obsdiag, para; site1=[xx, yy], site2=[xx - 1, yy + 1], get_op=get_op_Heisenberg)
        @reduce Eg += (get(Obs2sidiag_lurd, "hijNNN", NaN) + get(Obs2sidiag_ruld, "hijNNN", NaN))
        GC.gc()
    end
    GC.gc()
    magnetization = magnetization / (Lx * Ly)
    @show magnetization
    Eg = Eg / (Lx * Ly)
    @show Eg
end

main()