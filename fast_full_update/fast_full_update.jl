function fast_full_update!(ipeps::iPEPS, envs::iPEPSenv,)
    Lx = ipeps.Lx
    Ly = ipeps.Ly

    τlis = para[:τlis]
    Dk = para[:Dk]
    verbose = para[:verbose]
    NNNmethod = para[:NNNmethod]
    maxStep1τ = para[:maxStep1τ]
    hams = HamFunc(para)
    itsum = 0  # 记录总的迭代次数
    itime = 0.0  # 演化的总虚时间
    Ebefore = 0.0
    for τ in τlis
        gates = get_gates(hams, τ)
        for it in 0:maxStep1τ
            @time "FFU one step" errlis, prodNrm = fast_full_update_1step!(ipeps, envs, Dk, gates; verbose=verbose)
            itime += τ
            println("Truncation error = $(maximum(errlis)), total imaginary time = $itime")
            # ======== 检查能量收敛性 ====== See: PRB 104, 155118 (2021), Appendix 3.C
            E = -log(prodNrm) / τ
            println("Estimated energy per site is $(E / (Lx*Ly))")
            # =============================
            it += 1
            itsum += 1
            println("=========== Step τ=$τ, iteration $it, total iteration $itsum =======")
            println()
            # ======== 提前终止循环的情况 ===========
            if abs((E - Ebefore) / Ebefore) < para[:Etol] * τ^2
                Ebefore = E
                println("!! Energy converge. Reduce imaginary time step")
                break
            end
            Ebefore = E
            flush(stdout)
        end
    end
    return nothing
end


function fast_full_update_1step!(ipeps::iPEPS, envs::iPEPSenv, Dk::Int, gates::Vector{TensorMap}; verbose=1)
    Lx = ipeps.Lx
    Ly = ipeps.Ly
    errlis = Vector{Float64}(undef, 2 * length(gates) * Lx * Ly)  # 总的 bond 数
    prodNrm = one(Float64)
    Nb = one(Int)
    # ================= 最近邻相互作用 ==============
    # 逐行更新横向Bond
    for yy in 1:Ly, xx in 1:Lx
        err, nrm = bond_proj_lr!(ipeps, envs, xx, yy, Dk, gates[1])
        verbose > 1 ? println("横向更新 [$xx, $yy], error=$err") : nothing
        errlis[Nb] = err
        prodNrm *= nrm
        Nb += 1
    end
    # 逐列更新纵向Bond
    for xx in 1:Lx, yy in 1:Ly
        err, nrm = bond_proj_tb!(ipeps, envs, xx, yy, Dk, gates[1])
        verbose > 1 ? println("纵向更新 [$xx, $yy], error=$err") : nothing
        errlis[Nb] = err
        prodNrm *= nrm
        Nb += 1
    end
    # ============= TODO 次近邻相互作用 =============
    # if length(gates) >= 2
    #     for yy in 1:Ly, xx in 1:Lx
    #         errup1, errup2, errdn1, errdn2, nrmup1, nrmup2, nrmdn1, nrmdn2 = diag_proj_lu2rd!(ipeps, xx, yy, Dk, gates[2]; NNN=NNN)
    #         errlis[Nb] = maximum([errup1, errup2, errdn1, errdn2])
    #         verbose > 1 ? println("右下对角更新 [$xx, $yy], error=$(errlis[Nb])") : nothing
    #         prodNrm *= (nrmup1 * nrmup2 * nrmdn1 * nrmdn2)^2  # 次近邻是演化两次 τ/2, 因此这里要平方
    #         Nb += 1

    #         errup1, errup2, errdn1, errdn2, nrmup1, nrmup2, nrmdn1, nrmdn2 = diag_proj_ru2ld!(ipeps, xx, yy, Dk, gates[2]; NNN=NNN)
    #         errlis[Nb] = maximum([errup1, errup2, errdn1, errdn2])
    #         verbose > 1 ? println("左下对角更新 [$xx, $yy], error=$(errlis[Nb])") : nothing
    #         prodNrm *= (nrmup1 * nrmup2 * nrmdn1 * nrmdn2)^2
    #         Nb += 1
    #     end
    # end

    return errlis, prodNrm
end