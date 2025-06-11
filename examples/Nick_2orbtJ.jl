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

# 正方格子双层两轨道LNO327 t-J模型, Z₂charge × SU₂spin. 

include("../iPEPS_Fermionic/iPEPS.jl")
include("../CTMRG_Fermionic/CTMRG.jl")
include("../models/tJ_Z2SU2_2orb.jl")
include("../simple_update_Fermionic/simple_update_anisotropic.jl")
include("../Cal_Obs_Fermionic/Cal_Obs.jl")

function main()
    # ipepsγλ, para_not = load("/home/tcmp2/JuliaProjects/tJZ2_Lx4Ly4_t3.0t'0.51J1.0J'0.0289h0.6mu5.5_ipeps_D8.jld2", "ipeps", "para")

    para = Dict{Symbol,Any}()
    para[:tc] = 0.483
    para[:tperp] = 0.635
    para[:td] = 0.11
    para[:Jperp] = 0.4
    para[:JH] = 0.0
    para[:Jc] = 0.233  # not used now. Keep zero
    para[:V] = 0.239  # NN repusion
    para[:εx] = 0.367 - 0.3
    para[:εz] = 0.0 - 0.3
    para[:τlisSU] = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001]
    para[:minStep1τ] = 10   # 对每个虚时步长 τ , 最少投影这么多步
    para[:maxStep1τ] = 20  # 对每个虚时步长 τ , 最多投影这么多步
    para[:Dk] = 4  # Dkept in the simple udate
    para[:χ] = 130  # env bond dimension
    para[:CTMit] = 5  # Maximum CTMRG iteration times
    para[:CTMparallel] = true  # use parallel CTMRG or not. Use with MKL.
    para[:CTMthreshold] = 1e-12
    para[:Etol] = 1e-5  # simple update 能量差小于 para[:Etol]*τ² 这个数就可以继续增大步长. 1e-5对小size
    para[:verbose] = 1
    para[:TrotterOrder] = 2 # 用几阶Trotter分解,设为1或2

    para[:pspace] = Z₂SU₂tJ2orb.pspace
    pspace = Z₂SU₂tJ2orb.pspace
    aspacelr = Rep[ℤ₂×SU₂]((0, 0) => 1, (1, 1 // 2) => 1)
    aspacetb = Rep[ℤ₂×SU₂]((0, 0) => 1, (1, 1 // 2) => 1)

    Lx = 2
    Ly = 2
    # Dir on Window 
    dir = "D:/iPEPS_projects/LNO_2orb/data/LNO_Lx$(Lx)Ly$(Ly)_SU_V$(para[:V])JH$(para[:JH])Jperp$(para[:Jperp])_D$(para[:Dk])chi$(para[:χ])CTMit$(para[:CTMit])/"
    isdir(dir) ? nothing : mkdir(dir)
    # simple update
    ipepsγλ = iPEPSΓΛ(pspace, aspacelr, aspacetb, Lx, Ly; dtype=Float64)
    simple_update_aniso!(ipepsγλ, tJ2orb_hij, para)
    # save(ipepsγλ, para, "/home/tcmp2/JuliaProjects/tJZ2SU2_Lx$(Lx)Ly$(Ly)_t$(para[:t])tp$(para[:tp])J$(para[:J])Jp$(para[:Jp])V$(para[:V])mu$(para[:μ])_ipeps_D$(para[:Dk]).jld2")
    # ipepsγλ, para = load("/home/tcmp2/JuliaProjects/tJZ2_Lx$(Lx)Ly$(Ly)_t$(para[:t])tp$(para[:tp])J$(para[:J])Jp$(para[:Jp])mu$(para[:μ])_ipeps_D$(para[:Dk]).jld2", "ipeps", "para")

    # 转换为正常形式, 做 fast full update
    ipeps = iPEPS(ipepsγλ)
    @show space(ipeps[1, 1])
    ipepsbar = bar(ipeps)
    envs = iPEPSenv(ipeps)
    # check_qn(ipeps, envs)

    # 最后再做CTMRG
    CTMRG!(ipeps, ipepsbar, envs, para[:χ], para[:CTMit]; parallel=para[:CTMparallel], threshold=para[:CTMthreshold])
    # save(ipeps, envs, para, "/home/tcmp2/JuliaProjects/tJZ2SU2_Lx$(Lx)Ly$(Ly)_SU_t$(para[:t])tp$(para[:tp])J$(para[:J])Jp$(para[:Jp])V$(para[:V])mu$(para[:μ])_ipepsEnv_D$(para[:Dk])chi$(para[:χ]).jld2")
    GC.gc()
    # 计算观测量
    println("============== Calculating Obs ====================")
    site1Obs = ["Nx", "Nz", "Δₛx", "Δₛdagx", "Δₛz", "Δₛdagz"]   # 计算这些单点观测量

    rslt1s = Vector{Dict}(undef, Lx * Ly)
    rslt2s_h = Vector{Dict}(undef, Lx * Ly)
    rslt2s_v = Vector{Dict}(undef, Lx * Ly)

    fillingx = 0.0
    fillingz = 0.0
    Eg = 0.0
    # 线程数较多则不用parfor, 会占用过多内存
    for (ind, val) in enumerate(CartesianIndices((Lx, Ly)))
        (xx, yy) = Tuple(val)
        Obs1si = Cal_Obs_1site(ipeps, ipepsbar, envs, site1Obs, para; site=[xx, yy], get_op=get_op_tJ)
        @show (xx, yy, Obs1si)
        fillingx += get(Obs1si, "Nx", NaN)
        fillingz += get(Obs1si, "Nz", NaN)
        rslt1s[ind] = Obs1si

        Obs2si_h = Cal_Obs_2site(ipeps, ipepsbar, envs, ["hijNNx"], para; site1=[xx, yy], site2=[xx + 1, yy], get_op=get_op_tJ)
        @show (xx, yy, Obs2si_h)
        rslt2s_h[ind] = Obs2si_h
        Obs2si_v = Cal_Obs_2site(ipeps, ipepsbar, envs, ["hijNNy"], para; site1=[xx, yy], site2=[xx, yy + 1], get_op=get_op_tJ)
        @show (xx, yy, Obs2si_v)
        rslt2s_v[ind] = Obs2si_v
        Eg += (get(Obs2si_h, "hijNNx", NaN) + get(Obs2si_v, "hijNNy", NaN))

        GC.gc()
    end
    fillingx = fillingx / (Lx * Ly * 2)
    fillingz = fillingz / (Lx * Ly * 2)
    @show fillingx, fillingz
    Eg = Eg / (Lx * Ly) + fillingx * para[:εx] + fillingz * para[:εz]
    @show Eg

    # =================== save Obs to file ================================
    # Obsname = joinpath("/home/tcmp2/JuliaProjects/", "tJZ2SU2_Lx$(Lx)Ly$(Ly)_SU_t$(para[:t])tp$(para[:tp])J$(para[:J])Jp$(para[:Jp])V$(para[:V])mu$(para[:μ])_ipepsEnv_D$(para[:Dk])chi$(para[:χ])_Obs.h5")
    Obsname = joinpath(dir, "Obs.h5")
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
    finally
        close(f)
    end

    return nothing
end

main()