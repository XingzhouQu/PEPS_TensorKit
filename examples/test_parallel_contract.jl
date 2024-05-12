using JLD2
using Strided
using TensorKit
import TensorKit.×
using Base.Threads
include("../iPEPS_Fermionic/iPEPS.jl")
include("../CTMRG_Fermionic/CTMRG.jl")
include("../Cal_Obs_Fermionic/Cal_Obs.jl")
include("../models/Hubbard_Z2SU2.jl")
@show pkgversion(TensorKit)
# Strided.enable_threads()
@show nthreadpools()
@show Threads.nthreads()
# Strided.enable_threaded_mul()

function main()
    ipeps, envs, para = load("/home/tcmp2/JuliaProjects/Hubbard_t1.0U8.0mu4.0_ipepsEnv_D10chi200.jld2", "ipeps", "envs", "para")
    Lx = ipeps.Lx
    Ly = ipeps.Ly
    @show para[:χ]
    ipepsbar = bar(ipeps)

    println("============== Calculating Obs ====================")
    site1Obs = ["N"]                                # 计算这些单点观测量
    site2Obs = ["hij", "SS", "NN", "Δₛ", "Δₛdag"]   # 计算这些两点观测量
    # sites = [[x, y] for x in 1:Lx, y in 1:Ly]
    Eg = 0.0
    doping = 0.0
    # Threads.@threads 
    for ind in CartesianIndices((Lx, Ly))
        (xx, yy) = Tuple(ind)
        Obs1si = Cal_Obs_1site(ipeps, ipepsbar, envs, site1Obs, para; site=[xx, yy], get_op=get_op_Hubbard)
        @show Obs1si
        doping += get(Obs1si, "N", NaN)

        GC.gc()
        Obs2si = Cal_Obs_2site(ipeps, ipepsbar, envs, site2Obs, para; site1=[xx, yy], site2=[xx + 1, yy], get_op=get_op_Hubbard)
        @show Obs2si
        Eg += get(Obs2si, "hij", NaN)
        GC.gc()

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