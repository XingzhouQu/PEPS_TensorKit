using JLD2
using TensorKit

function genScript(para::Dict{Symbol,Any})
    jobname = string("Obs", para[:jobname])
    ncpu = para[:ncpu]
    nthreads = para[:nthreads]
    mem = para[:mem]
    parpath = joinpath(para[:RsltFldr], "para.jld2")
    str = string(
        "#!/bin/bash
        #BSUB -J $jobname
        #BSUB -q score
        #BSUB -n $ncpu
        #BSUB -R 'rusage[mem=$mem]'
        #BSUB -R span[hosts=1]
        #BSUB -o logfiles/ObsiPEPStJ%J.log -e logfiles/%J.err
        echo 'Job Started at'
        date
        MKL_NUM_THREADS=$ncpu /public/home/users/ucas001c/julia1.10/julia-1.10.3/bin/julia --project=/public/home/users/ucas001c/julia1.10/jl20240601 --heap-size-hint=90G main_Obs.jl $parpath
        echo 'Job finished at'
        date")

    file = open("./Obs.lsf", "w")
    write(file, str)
    close(file)

    run(`chmod 777 Obs.lsf`)
    return nothing
end

function setPara(Lx, Ly, t, tp, J, Jp, h, μ)
    # Generate the Para struct.
    println("Generating Para....")

    para = Dict{Symbol,Any}()
    # model parameters
    para[:Lx] = Lx
    para[:Ly] = Ly
    para[:t] = t
    para[:tp] = tp
    para[:J] = J
    para[:Jp] = Jp
    para[:h] = h
    para[:μ] = μ
    para[:τlisSU] = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001]
    para[:τlisFFU] = [0.01, 0.005, 0.001, 0.0001]
    para[:minStep1τ] = 50   # 对每个虚时步长 τ , 最少投影这么多步
    para[:maxStep1τ] = 200  # 对每个虚时步长 τ , 最多投影这么多步
    para[:maxiterFFU] = 60
    para[:tolFFU] = 1e-10  # FFU 中损失函数的 Tolerence
    para[:Dk] = 6  # Dkept in the simple udate
    para[:χ] = 200  # env bond dimension
    para[:CTMit] = 20  # CTMRG iteration times
    para[:CTMparallel] = false  # use parallel CTMRG or not. Use with MKL.
    para[:Etol] = 1e-6  # simple update 能量差小于 para[:Etol]*τ² 这个数就可以继续增大步长. 1e-5对小size
    para[:verbose] = 1
    para[:NNNmethod] = :bond
    para[:pspace] = Rep[ℤ₂](0 => 1, 1 => 2)
    # HPC parameters
    para[:saveiPEPS] = true
    para[:saveEnv] = true
    para[:useexist] = false
    para[:jobname] = "tJZ2_Lx$(Lx)Ly$(Ly)_SU_t$(t)tp$(tp)J$(J)Jp$(Jp)h$(h)mu$(μ)_D$(para[:Dk])chi$(para[:χ])CTMit$(para[:CTMit])"
    para[:oldiPEPSDir] = string("/public/home/users/ucas001c/xzqu/iPEPS_ttpJ/iPEPSstorage/", para[:jobname])
    para[:RsltFldr] = string("/public/home/users/ucas001c/xzqu/iPEPS_ttpJ/Rslt/", para[:jobname])
    para[:iPEPSDir] = string("/public/home/users/ucas001c/xzqu/iPEPS_ttpJ/iPEPSstorage/", para[:jobname])
    para[:ncpu] = 16
    para[:mem] = 90000
    para[:nthreads] = 16

    return para
end

function main()
    tlis = [3.0]
    tplis = [0.51]
    Jlis = [1.0]
    Jplis = [0.0289]
    hlis = [0.6]
    mulis = [5.2]
    Lx = 4
    Ly = 4
    for t in tlis, tp in tplis, J in Jlis, Jp in Jplis, h in hlis, mu in mulis
        para = setPara(Lx, Ly, t, tp, J, Jp, h, mu)
        genScript(para)
        for (key, value) in para
            println("$key => $value")
        end
        println()
        println("Please submit. bsub < Obs.lsf")
        sleep(10)
    end
    return nothing
end

main()