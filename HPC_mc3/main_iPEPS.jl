using MKL
using LinearAlgebra
using TensorOperations, TensorKit
import TensorKit.×
using Statistics
using JLD2, HDF5
using Strided, FLoops
# 测试正方格子 t-t'-J 模型加 Zeeman 场, Z₂charge. 

include("/public/home/users/ucas001c/xzqu/iPEPS_ttpJ/code/iPEPS_Fermionic/iPEPS.jl")
include("/public/home/users/ucas001c/xzqu/iPEPS_ttpJ/code/CTMRG_Fermionic/CTMRG.jl")
include("/public/home/users/ucas001c/xzqu/iPEPS_ttpJ/code/models/tJ_Z2.jl")
include("/public/home/users/ucas001c/xzqu/iPEPS_ttpJ/code/simple_update_Fermionic/simple_update.jl")
include("/public/home/users/ucas001c/xzqu/iPEPS_ttpJ/code/fast_full_update_Fermionic/fast_full_update.jl")
include("/public/home/users/ucas001c/xzqu/iPEPS_ttpJ/code/Cal_Obs_Fermionic/Cal_Obs.jl")

function mainiPEPS(para)
    pspace = para[:pspace]
    aspacelr = Rep[ℤ₂](0 => 1, 1 => 2)
    aspacetb = Rep[ℤ₂](0 => 1, 1 => 2)
    Lx = para[:Lx]
    Ly = para[:Ly]
    if para[:useexist]
        ipepsγλ, _ = load(joinpath(para[:oldiPEPSDir], "ipeps_D$(para[:Dk]).jld2"), "ipeps", "para")
    else
        ipepsγλ = iPEPSΓΛ(pspace, aspacelr, aspacetb, Lx, Ly; dtype=Float64)
    end

    # simple update
    println("============== Simple update ====================")
    simple_update!(ipepsγλ, tJ_hij, para)
    if para[:saveiPEPS]
        save(ipepsγλ, para, joinpath(para[:iPEPSDir], "ipeps_D$(para[:Dk]).jld2"))
    else
        println("Do not save iPEPS (ΓΛ form)")
    end

    # 转换为正常形式,
    ipeps = iPEPS(ipepsγλ)
    @show space(ipeps[1, 1])
    ipepsbar = bar(ipeps)
    envs = iPEPSenv(ipeps)
    # check_qn(ipeps, envs)
    # CTMRG!(ipeps, ipepsbar, envs, para[:χ], 2)
    # fast_full_update!(ipeps, envs, Hubbard_hij, para)
    # check_qn(ipeps, envs)

    # 最后再做CTMRG
    println("============== CTMRG ====================")
    CTMRG!(ipeps, ipepsbar, envs, para[:χ], para[:CTMit]; parallel=para[:CTMparallel])
    if para[:saveEnv]
        save(ipeps, envs, para, joinpath(para[:iPEPSDir], "ipepsEnv_D$(para[:Dk])chi$(para[:χ]).jld2"))
    else
        println("Do not save iPEPS Env")
    end
    GC.gc()

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
    # rslt = Dict(Gates[ind] => (vals[ind] / nrm) for ind in 1:length(vals))
    @floop for (ind, val) in enumerate(CartesianIndices((Lx, Ly)))
        (xx, yy) = Tuple(val)
        Obs1si = Cal_Obs_1site(ipeps, ipepsbar, envs, site1Obs, para; site=[xx, yy], get_op=get_op_tJ)
        @show (xx, yy, Obs1si)
        @reduce filling += get(Obs1si, "N", NaN)
        @reduce magnetization += abs(get(Obs1si, "Sz", NaN))
        rslt1s[ind] = Obs1si

        Obs2si_h = Cal_Obs_2site(ipeps, ipepsbar, envs, site2Obs, para; site1=[xx, yy], site2=[xx + 1, yy], get_op=get_op_tJ)
        @show (xx, yy, Obs2si_h)
        rslt2s_h[ind] = Obs2si_h
        Obs2si_v = Cal_Obs_2site(ipeps, ipepsbar, envs, site2Obs, para; site1=[xx, yy], site2=[xx, yy + 1], get_op=get_op_tJ)
        @show (xx, yy, Obs2si_v)
        rslt2s_v[ind] = Obs2si_v
        @reduce Eg += (get(Obs2si_h, "hijNN", NaN) + get(Obs2si_v, "hijNN", NaN))

        Obs2sidiag_lurd = Cal_Obs_2site(ipeps, ipepsbar, envs, site2Obsdiag, para; site1=[xx, yy], site2=[xx + 1, yy + 1], get_op=get_op_tJ)
        rslt2s_lu2rd[ind] = Obs2sidiag_lurd
        Obs2sidiag_ruld = Cal_Obs_2site(ipeps, ipepsbar, envs, site2Obsdiag, para; site1=[xx, yy], site2=[xx - 1, yy + 1], get_op=get_op_tJ)
        rslt2s_ru2ld[ind] = Obs2sidiag_ruld
        @reduce Eg += (get(Obs2sidiag_lurd, "hijNNN", NaN) + get(Obs2sidiag_ruld, "hijNNN", NaN))
        GC.gc()
    end
    filling = filling / (Lx * Ly)
    @show filling
    magnetization = magnetization / (Lx * Ly)
    @show magnetization
    Eg = Eg / (Lx * Ly) + filling * para[:μ]
    @show Eg

    # =================== save Obs to file ================================
    Obsname = joinpath(para[:RsltFldr], "Obs.h5")
    f = h5open(Obsname, "w")
    T = eltype(ipeps[1, 1])
    try
        for obs in site1Obs  # eg: obs == "N"
            tmp = Matrix{T}(undef, Lx * Ly, 3)  # 注意！！这里必须是默认数据类型才能存进去. 如涉及虚数观测量, 坐标会存虚数.
            for (ind, val) in enumerate(CartesianIndices((Lx, Ly)))
                (xx, yy) = Tuple(val)
                for Obs1si in rslt1s  # Obs1si is a Dict. 记录在某个site的所有单点观测量值
                    tmp[ind, 1], tmp[ind, 2] = xx, yy
                    tmp[ind, 3] = get(Obs1si, obs, NaN)
                end
            end
            write(f, obs, sortslices(tmp, dims=1))
        end
        for obs in site2Obs  # eg: obs == "NN"
            tmp_h = Matrix{T}(undef, Lx * Ly, 5)
            tmp_v = Matrix{T}(undef, Lx * Ly, 5)
            for (ind, val) in enumerate(CartesianIndices((Lx, Ly)))
                (xx, yy) = Tuple(val)
                for Obs2si_h in rslt2s_h  # Obs2si is a Dict. 记录在某个site的所有两点观测量值
                    tmp_h[ind, 1], tmp_h[ind, 2], tmp_h[ind, 3], tmp_h[ind, 4] = xx, yy, xx + 1 - Int(ceil((xx + 1) / Lx) - 1) * Lx, yy - Int(ceil(yy / Ly) - 1) * Ly
                    tmp_h[ind, 5] = get(Obs2si_h, obs, NaN)
                end
                for Obs2si_v in rslt2s_v  # Obs2si is a Dict. 记录在某个site的所有两点观测量值
                    tmp_v[ind, 1], tmp_v[ind, 2], tmp_v[ind, 3], tmp_v[ind, 4] = xx, yy, xx - Int(ceil(xx / Lx) - 1) * Lx, yy + 1 - Int(ceil((yy + 1) / Ly) - 1) * Ly
                    tmp_v[ind, 5] = get(Obs2si_v, obs, NaN)
                end
            end
            write(f, obs, sortslices(vcat(tmp_h, tmp_v), dims=1))
        end
        for obs in site2Obsdiag  # eg: obs == "NN"
            tmp_lurd = Matrix{T}(undef, Lx * Ly, 5)
            tmp_ruld = Matrix{T}(undef, Lx * Ly, 5)
            for (ind, val) in enumerate(CartesianIndices((Lx, Ly)))
                (xx, yy) = Tuple(val)
                for Obs2si_lurd in rslt2s_lu2rd  # Obs2si is a Dict. 记录在某个site的所有两点观测量值
                    tmp_lurd[ind, 1], tmp_lurd[ind, 2], tmp_lurd[ind, 3], tmp_lurd[ind, 4] = xx, yy, xx + 1 - Int(ceil((xx + 1) / Lx) - 1) * Lx, yy + 1 - Int(ceil((yy + 1) / Ly) - 1) * Ly
                    tmp_lurd[ind, 5] = get(Obs2si_lurd, obs, NaN)
                end
                for Obs2si_ruld in rslt2s_ru2ld  # Obs2si is a Dict. 记录在某个site的所有两点观测量值
                    tmp_ruld[ind, 1], tmp_ruld[ind, 2], tmp_ruld[ind, 3], tmp_ruld[ind, 4] = xx, yy, xx - 1 - Int(ceil((xx - 1) / Lx) - 1) * Lx, yy + 1 - Int(ceil((yy + 1) / Ly) - 1) * Ly
                    tmp_ruld[ind, 5] = get(Obs2si_ruld, obs, NaN)
                end
            end
            write(f, string(obs, "diag"), sortslices(vcat(tmp_lurd, tmp_ruld), dims=1))
        end
    finally
        close(f)
    end
    return nothing
end


function loadPara_Run(parpath)
    # Accept the path of para file, read it and run iPEPS
    @load parpath para
    #   global para = para
    for (key, value) in para
        println("$key => $value")
    end
    println()

    # Set the multithreading environment.
    LinearAlgebra.BLAS.set_num_threads(para[:nthreads])
    Strided.enable_threads()
    @show Threads.nthreadpools()
    @show Threads.nthreads()
    @show Sys.CPU_THREADS
    # Run iPEPS
    mainiPEPS(para)

    return nothing
end

loadPara_Run(ARGS[1])
