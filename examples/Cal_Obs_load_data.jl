using MKL
using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(8)
using TensorOperations, TensorKit
using Statistics
import TensorKit.×
using JLD2, HDF5
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
    # ipepsγλ, para_not = load("/home/tcmp2/JuliaProjects/tJZ2_Lx4Ly4_t3.0t'0.51J1.0J'0.0289h0.6mu5.5_ipeps_D8.jld2", "ipeps", "para")

    Lx = 32
    Ly = 2
    ipeps, envs, para = load("/home/tcmp2/JuliaProjects/tJZ2_Lx32Ly2_SU_t3.0t'0.51J1.0J'0.0289h0.6mu5.2_ipepsEnv_D6chi300.jld2", "ipeps", "envs", "para")
    ipepsbar = bar(ipeps)
    # 计算观测量
    println("============== Calculating Obs ====================")
    site1Obs = ["N", "Sz"]             # 计算这些单点观测量
    site2Obs = ["hijNN", "SpSm", "SzSz", "NN", "Δₛ", "Δₛdag"]   # 计算这些两点观测量
    site2Obsdiag = ["hijNNN", "SpSm", "SzSz", "NN"]

    rslt1s = Vector{Dict}(undef, Lx * Ly)
    rslt2s_h = Vector{Dict}(undef, Lx * Ly)
    rslt2s_v = Vector{Dict}(undef, Lx * Ly)
    rslt2s_lu2rd = Vector{Dict}(undef, Lx * Ly)
    rslt2s_ru2ld = Vector{Dict}(undef, Lx * Ly)

    filling = 0.0
    magnetization = 0.0
    Eg = 0.0
    for (ind, val) in enumerate(CartesianIndices((Lx, Ly)))
        (xx, yy) = Tuple(val)
        Obs1si = Cal_Obs_1site(ipeps, ipepsbar, envs, site1Obs, para; site=[xx, yy], get_op=get_op_tJ)
        @show (xx, yy, Obs1si)
        filling += get(Obs1si, "N", NaN)
        magnetization += abs(get(Obs1si, "Sz", NaN))
        rslt1s[ind] = Obs1si

        Obs2si_h = Cal_Obs_2site(ipeps, ipepsbar, envs, site2Obs, para; site1=[xx, yy], site2=[xx + 1, yy], get_op=get_op_tJ)
        @show (xx, yy, Obs2si_h)
        rslt2s_h[ind] = Obs2si_h
        Obs2si_v = Cal_Obs_2site(ipeps, ipepsbar, envs, site2Obs, para; site1=[xx, yy], site2=[xx, yy + 1], get_op=get_op_tJ)
        @show (xx, yy, Obs2si_v)
        rslt2s_v[ind] = Obs2si_v

        Eg += (get(Obs2si_h, "hijNN", NaN) + get(Obs2si_v, "hijNN", NaN))
        Obs2sidiag_lurd = Cal_Obs_2site(ipeps, ipepsbar, envs, site2Obsdiag, para; site1=[xx, yy], site2=[xx + 1, yy + 1], get_op=get_op_tJ)
        rslt2s_lu2rd[ind] = Obs2sidiag_lurd
        Obs2sidiag_ruld = Cal_Obs_2site(ipeps, ipepsbar, envs, site2Obsdiag, para; site1=[xx, yy], site2=[xx - 1, yy + 1], get_op=get_op_tJ)
        rslt2s_ru2ld[ind] = Obs2sidiag_ruld
        Eg += (get(Obs2sidiag_lurd, "hijNNN", NaN) + get(Obs2sidiag_ruld, "hijNNN", NaN))
        flush(stdout)
    end
    filling = filling / (Lx * Ly)
    @show filling
    magnetization = magnetization / (Lx * Ly)
    @show magnetization
    Eg = Eg / (Lx * Ly) + filling * para[:μ]
    @show Eg

    # =================== save Obs to file ================================
    Obsname = joinpath("/home/tcmp2/JuliaProjects/", "tJZ2_Lx$(Lx)Ly$(Ly)_SU_t$(para[:t])t'$(para[:tp])J$(para[:J])J'$(para[:Jp])h$(para[:h])mu$(para[:μ])_ipepsEnv_D$(para[:Dk])chi$(para[:χ])_Obs.h5")
    f = h5open(Obsname, "w")
    T = eltype(ipeps[1, 1])
    try
        for obs in site1Obs  # eg: obs == "N"
            tmp = Matrix{T}(undef, Lx * Ly, 3)  # 注意！！这里必须是默认数据类型才能存进去. 如涉及虚数观测量, 坐标会存虚数.
            for (ind, val) in enumerate(CartesianIndices((Lx, Ly)))
                (xx, yy) = Tuple(val)
                tmp[ind, 1], tmp[ind, 2] = xx, yy
                tmp[ind, 3] = get(rslt1s[ind], obs, NaN)
            end
            write(f, obs, sortslices(tmp, dims=1))
        end
        for obs in site2Obs  # eg: obs == "NN"
            tmp_h = Matrix{T}(undef, Lx * Ly, 5)
            tmp_v = Matrix{T}(undef, Lx * Ly, 5)
            for (ind, val) in enumerate(CartesianIndices((Lx, Ly)))
                (xx, yy) = Tuple(val)
                tmp_h[ind, 1], tmp_h[ind, 2], tmp_h[ind, 3], tmp_h[ind, 4] = xx, yy, xx + 1 - Int(ceil((xx + 1) / Lx) - 1) * Lx, yy - Int(ceil(yy / Ly) - 1) * Ly
                tmp_h[ind, 5] = get(rslt2s_h[ind], obs, NaN)
                tmp_v[ind, 1], tmp_v[ind, 2], tmp_v[ind, 3], tmp_v[ind, 4] = xx, yy, xx - Int(ceil(xx / Lx) - 1) * Lx, yy + 1 - Int(ceil((yy + 1) / Ly) - 1) * Ly
                tmp_v[ind, 5] = get(rslt2s_v[ind], obs, NaN)
            end
            write(f, obs, sortslices(vcat(tmp_h, tmp_v), dims=1))
        end
        for obs in site2Obsdiag  # eg: obs == "NN"
            tmp_lurd = Matrix{T}(undef, Lx * Ly, 5)
            tmp_ruld = Matrix{T}(undef, Lx * Ly, 5)
            for (ind, val) in enumerate(CartesianIndices((Lx, Ly)))
                (xx, yy) = Tuple(val)
                tmp_lurd[ind, 1], tmp_lurd[ind, 2], tmp_lurd[ind, 3], tmp_lurd[ind, 4] = xx, yy, xx + 1 - Int(ceil((xx + 1) / Lx) - 1) * Lx, yy + 1 - Int(ceil((yy + 1) / Ly) - 1) * Ly
                tmp_lurd[ind, 5] = get(rslt2s_lu2rd[ind], obs, NaN)
                tmp_ruld[ind, 1], tmp_ruld[ind, 2], tmp_ruld[ind, 3], tmp_ruld[ind, 4] = xx, yy, xx - 1 - Int(ceil((xx - 1) / Lx) - 1) * Lx, yy + 1 - Int(ceil((yy + 1) / Ly) - 1) * Ly
                tmp_ruld[ind, 5] = get(rslt2s_ru2ld[ind], obs, NaN)
            end
            write(f, string(obs, "diag"), sortslices(vcat(tmp_lurd, tmp_ruld), dims=1))
        end
    finally
        close(f)
    end

    return nothing
end

main()