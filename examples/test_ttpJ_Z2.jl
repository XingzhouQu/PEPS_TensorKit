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

    para = Dict{Symbol,Any}()
    para[:t] = 3.0
    para[:tp] = 0.51
    para[:J] = 1.0
    para[:Jp] = 0.0289
    para[:h] = 0.6
    para[:μ] = 5.2  # set μ = 5.2,  n = 0.89986
    para[:τlisSU] = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001]
    para[:τlisFFU] = [0.01, 0.005, 0.001, 0.0001]
    para[:minStep1τ] = 50   # 对每个虚时步长 τ , 最少投影这么多步
    para[:maxStep1τ] = 500  # 对每个虚时步长 τ , 最多投影这么多步
    para[:maxiterFFU] = 60
    para[:tolFFU] = 1e-10  # FFU 中损失函数的 Tolerence
    para[:Dk] = 6  # Dkept in the simple udate
    para[:χ] = 300  # env bond dimension
    para[:CTMit] = 30  # CTMRG iteration times
    para[:CTMparallel] = false  # use parallel CTMRG or not. Use with MKL.
    para[:Etol] = 1e-6  # simple update 能量差小于 para[:Etol]*τ² 这个数就可以继续增大步长. 1e-5对小size
    para[:verbose] = 1
    para[:NNNmethod] = :bond
    para[:pspace] = Rep[ℤ₂](0 => 1, 1 => 2)

    pspace = Rep[ℤ₂](0 => 1, 1 => 2)
    aspacelr = Rep[ℤ₂](0 => 1, 1 => 2)
    aspacetb = Rep[ℤ₂](0 => 1, 1 => 2)
    Lx = 32
    Ly = 2
    # # 决定初态每条腿的量子数
    # aspacel = Matrix{GradedSpace}(undef, Lx, Ly)
    # aspacet = Matrix{GradedSpace}(undef, Lx, Ly)
    # aspacer = Matrix{GradedSpace}(undef, Lx, Ly)
    # aspaceb = Matrix{GradedSpace}(undef, Lx, Ly)
    # # TODO: 合理选择每条腿的量子数，达到固定的掺杂
    # ipepsγλ = iPEPSΓΛ(pspace, aspacel, aspacet, aspacer, aspaceb, Lx, Ly; dtype=Float64)

    # simple update
    # ipepsγλ = iPEPSΓΛ(pspace, aspacelr, aspacetb, Lx, Ly; dtype=Float64)
    # simple_update!(ipepsγλ, tJ_hij, para)
    # save(ipepsγλ, para, "/home/tcmp2/JuliaProjects/tJZ2_Lx$(Lx)Ly$(Ly)_t$(para[:t])t'$(para[:tp])J$(para[:J])J'$(para[:Jp])h$(para[:h])mu$(para[:μ])_ipeps_D$(para[:Dk]).jld2")
    ipepsγλ, para = load("/home/tcmp2/JuliaProjects/tJZ2_Lx$(Lx)Ly$(Ly)_t$(para[:t])t'$(para[:tp])J$(para[:J])J'$(para[:Jp])h$(para[:h])mu$(para[:μ])_ipeps_D$(para[:Dk]).jld2", "ipeps", "para")

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
    save(ipeps, envs, para, "/home/tcmp2/JuliaProjects/tJZ2_Lx$(Lx)Ly$(Ly)_SU_t$(para[:t])t'$(para[:tp])J$(para[:J])J'$(para[:Jp])h$(para[:h])mu$(para[:μ])_ipepsEnv_D$(para[:Dk])chi$(para[:χ]).jld2")
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
    Obsname = joinpath("/home/tcmp2/JuliaProjects/", "tJZ2_Lx$(Lx)Ly$(Ly)_SU_t$(para[:t])t'$(para[:tp])J$(para[:J])J'$(para[:Jp])h$(para[:h])mu$(para[:μ])_ipepsEnv_D$(para[:Dk])chi$(para[:χ])_Obs.h5")
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

main()