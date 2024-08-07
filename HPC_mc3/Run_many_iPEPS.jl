using JLD2
using TensorKit

function genScript(para::Dict{Symbol,Any})
    jobname = para[:jobname]
    ncpu = para[:ncpu]
    nthreads = para[:nthreads]
    mem = para[:mem]
    parpath = joinpath(para[:RsltFldr], "para.jld2")
    @save parpath para
    if para[:saveiPEPS]
        tmppath = joinpath(para[:iPEPSDir], "para.jld2")
        @save tmppath para
    end
    jobname = para[:jobname]
    str = string(
        "#!/bin/bash
        #BSUB -J $jobname
        #BSUB -q score
        #BSUB -n $ncpu
        #BSUB -R 'rusage[mem=$mem]'
        #BSUB -R span[hosts=1]
        #BSUB -o logfiles/iPEPStJ%J.log -e logfiles/%J.err
        echo 'Job Started at'
        date
        MKL_NUM_THREADS=$ncpu /public/home/users/ucas001c/julia1.10/julia-1.10.3/bin/julia --project=/public/home/users/ucas001c/julia1.10/jl20240601 --heap-size-hint=90G main_tp_iPEPS.jl $parpath
        echo 'Job finished at'
        date")

    file = open("./iPEPS.lsf", "w")
    write(file, str)
    close(file)

    run(`chmod 777 iPEPS.lsf`)
    return nothing
end

function setPara(Lx, Ly, t, tp, J, Jp, μ)
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
    para[:μ] = μ
    para[:pspace] = Rep[ℤ₂×U₁]((0, 0) => 1, (1, 1 / 2) => 1, (1, -1 / 2) => 1)
    # update parameters
    para[:τlisSU] = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001]
    para[:τlisFFU] = [0.01, 0.005, 0.001, 0.0001]
    para[:minStep1τ] = 10   # 对每个虚时步长 τ , 最少投影这么多步
    para[:maxStep1τ] = 2000  # 对每个虚时步长 τ , 最多投影这么多步
    para[:maxiterFFU] = 60
    para[:tolFFU] = 1e-10  # FFU 中损失函数的 Tolerence
    para[:Dk] = 8  # Dkept in the simple udate
    para[:Etol] = 1e-5  # simple update 能量差小于 para[:Etol]*τ² 这个数就可以继续增大步长. 1e-5对小size
    para[:TrotterOrder] = 2 # 用几阶Trotter分解,设为1或2
    # CTMRG parameters
    para[:χ] = 250  # env bond dimension
    para[:CTMit] = 100  # CTMRG iteration times
    para[:CTMparallel] = false  # use parallel CTMRG or not. Use with MKL.
    para[:CTMthreshold] = 1e-12
    para[:verbose] = 1
    # HPC parameters
    para[:saveiPEPS] = true
    para[:saveEnv] = true
    para[:useexist] = false
    para[:jobname] = "tJZ2U1_Lx$(Lx)Ly$(Ly)_SU_t$(t)tp$(tp)J$(J)Jp$(Jp)mu$(μ)_D$(para[:Dk])chi$(para[:χ])CTMit$(para[:CTMit])"
    para[:oldiPEPSDir] = string("/public/home/users/ucas001c/xzqu/iPEPS_ttpJ/iPEPSstorage/", para[:jobname])
    para[:RsltFldr] = string("/public/home/users/ucas001c/xzqu/iPEPS_ttpJ/Rslt/", para[:jobname])
    para[:iPEPSDir] = string("/public/home/users/ucas001c/xzqu/iPEPS_ttpJ/iPEPSstorage/", para[:jobname])
    para[:ncpu] = 16
    para[:mem] = 90000
    para[:nthreads] = 16

    isdir(para[:RsltFldr]) ? nothing : mkdir(para[:RsltFldr])
    if para[:saveiPEPS]
        isdir(para[:iPEPSDir]) ? nothing : mkdir(para[:iPEPSDir])
    end
    return para
end

function main()
    tlis = [0.3]
    tplis = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
    Jlis = [1.0]
    Jplis = [0.0]
    mulis = [3.0, 2.0]
    Lx = 4
    Ly = 4
    for t in tlis, tp in tplis, J in Jlis, Jp in Jplis, mu in mulis
        para = setPara(Lx, Ly, t, tp, J, Jp, mu)
        genScript(para)
        for (key, value) in para
            println("$key => $value")
        end
        println()
        println("Please submit. bsub < iPEPS.lsf")
        sleep(10)
    end
    return nothing
end

main()