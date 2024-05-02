# TODO: add swap_gate
include("../iPEPS_Fermionic/swap_gate.jl")
include("./SU_gate_proj.jl")

function simple_update!(ipeps::iPEPSΓΛ, HamFunc::Function, para::Dict{Symbol,Any})
    Lx = ipeps.Lx
    Ly = ipeps.Ly

    τlis = para[:τlisSU]
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
            @time "SU one step" errlis, prodNrm = simple_update_1step!(ipeps, Dk, gates; verbose=verbose, NNN=NNNmethod)
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


"""
Generate trotter gates for simple update.\n
return a vector. [gateNN, gateNNN], etc. \n 
Notice: For Next-Nearest-Neighbor interaction, return √gateNNN and employ two paths in the following steps.
"""
function get_gates(hams::Vector{T}, τ::Number) where {T<:TensorMap}
    if length(hams) >= 3
        error("Only support up to next-nearest-neighbor interaction.")
    end
    gates = Vector{TensorMap}(undef, length(hams))
    gates[1] = exp(-τ * hams[1])
    # 注意次近邻哈密顿量要用两个对角路径求平均，用√gate
    length(hams) > 1 ? gates[2] = exp(-τ * hams[2] / 2) : nothing
    return gates
end


function simple_update_1step!(ipeps::iPEPSΓΛ, Dk::Int, gates::Vector{TensorMap}; verbose=1, NNN=:bond)
    Lx = ipeps.Lx
    Ly = ipeps.Ly
    errlis = Vector{Float64}(undef, 2 * length(gates) * Lx * Ly)  # 总的 bond 数
    prodNrm = one(Float64)
    Nb = one(Int)
    # ================= 最近邻相互作用 ==============
    # 逐行更新横向Bond
    for yy in 1:Ly, xx in 1:Lx
        err, nrm = bond_proj_lr!(ipeps, xx, yy, Dk, gates[1])
        verbose > 1 ? println("横向更新 [$xx, $yy], error=$err") : nothing
        errlis[Nb] = err
        prodNrm *= nrm
        Nb += 1
    end
    # 逐列更新纵向Bond
    for xx in 1:Lx, yy in 1:Ly
        err, nrm = bond_proj_tb!(ipeps, xx, yy, Dk, gates[1])
        verbose > 1 ? println("纵向更新 [$xx, $yy], error=$err") : nothing
        errlis[Nb] = err
        prodNrm *= nrm
        Nb += 1
    end
    # ============= 次近邻相互作用 =============
    if length(gates) >= 2
        for yy in 1:Ly, xx in 1:Lx
            errup1, errup2, errdn1, errdn2, nrmup1, nrmup2, nrmdn1, nrmdn2 = diag_proj_lu2rd!(ipeps, xx, yy, Dk, gates[2]; NNN=NNN)
            errlis[Nb] = maximum([errup1, errup2, errdn1, errdn2])
            verbose > 1 ? println("右下对角更新 [$xx, $yy], error=$(errlis[Nb])") : nothing
            prodNrm *= (nrmup1 * nrmup2 * nrmdn1 * nrmdn2)^2  # 次近邻是演化两次 τ/2, 因此这里要平方
            Nb += 1

            errup1, errup2, errdn1, errdn2, nrmup1, nrmup2, nrmdn1, nrmdn2 = diag_proj_ru2ld!(ipeps, xx, yy, Dk, gates[2]; NNN=NNN)
            errlis[Nb] = maximum([errup1, errup2, errdn1, errdn2])
            verbose > 1 ? println("左下对角更新 [$xx, $yy], error=$(errlis[Nb])") : nothing
            prodNrm *= (nrmup1 * nrmup2 * nrmdn1 * nrmdn2)^2
            Nb += 1
        end
    end

    return errlis, prodNrm
end


function diag_proj_lu2rd!(ipeps::iPEPSΓΛ, xx::Int, yy::Int, Dk::Int, gateNNN::TensorMap; NNN=:bond)
    if NNN == :bond
        errup1, errup2, nrmup1, nrmup2 = bond_proj_lu2rd_upPath!(ipeps, xx, yy, Dk, gateNNN)
        errdn1, errdn2, nrmdn1, nrmdn2 = bond_proj_lu2rd_dnPath!(ipeps, xx, yy, Dk, gateNNN)
    elseif NNN == :site
        errup1, errup2, nrmup1, nrmup2 = site_proj_lu2rd_upPath!(ipeps, xx, yy, Dk, gateNNN)
        errdn1, errdn2, nrmdn1, nrmdn2 = site_proj_lu2rd_dnPath!(ipeps, xx, yy, Dk, gateNNN)
    else
        error("NNN update method can only be `:bond` or `:site`")
    end
    return errup1, errup2, errdn1, errdn2, nrmup1, nrmup2, nrmdn1, nrmdn2
end


function diag_proj_ru2ld!(ipeps::iPEPSΓΛ, xx::Int, yy::Int, Dk::Int, gateNNN::TensorMap; NNN=:bond)
    if NNN == :bond
        errup1, errup2, nrmup1, nrmup2 = bond_proj_ru2ld_upPath!(ipeps, xx, yy, Dk, gateNNN)
        errdn1, errdn2, nrmdn1, nrmdn2 = bond_proj_ru2ld_dnPath!(ipeps, xx, yy, Dk, gateNNN)
    elseif NNN == :site
        errup1, errup2, nrmup1, nrmup2 = site_proj_ru2ld_upPath!(ipeps, xx, yy, Dk, gateNNN)
        errdn1, errdn2, nrmdn1, nrmdn2 = site_proj_ru2ld_dnPath!(ipeps, xx, yy, Dk, gateNNN)
    else
        error("NNN update method can only be `:bond` or `:site`")
    end
    return errup1, errup2, errdn1, errdn2, nrmup1, nrmup2, nrmdn1, nrmdn2
end