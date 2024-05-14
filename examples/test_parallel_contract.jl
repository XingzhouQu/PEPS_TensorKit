using JLD2
using Strided
using TensorKit
import TensorKit.×
using FLoops
using TensorOperations
using BenchmarkTools
include("../iPEPS_Fermionic/iPEPS.jl")
include("../CTMRG_Fermionic/CTMRG.jl")
include("../Cal_Obs_Fermionic/Cal_Obs.jl")
include("../models/Hubbard_Z2SU2.jl")
@show pkgversion(TensorKit)
Strided.enable_threads()
@show Threads.nthreadpools()
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
    Eg = Eg / (Lx * Ly)
    @show Eg
    doping = doping / (Lx * Ly)
    @show doping
    return nothing
end

@benchmark main()