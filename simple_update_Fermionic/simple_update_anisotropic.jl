"""
    isotropic simple_update with `gateNNx` and `gateNNy`.
"""

include("../iPEPS_Fermionic/swap_gate.jl")
include("./SU_gate_proj.jl")

function simple_update_aniso!(ipeps::iPEPSΓΛ, HamFunc::Function, para::Dict{Symbol,Any})
    Lx = ipeps.Lx::Int
    Ly = ipeps.Ly::Int

    τlis = para[:τlisSU]
    Dk = para[:Dk]
    verbose = para[:verbose]
    maxStep1τ = para[:maxStep1τ]
    TrotterOrder = para[:TrotterOrder]
    hams = HamFunc(para)
    itsum = 0  # 记录总的迭代次数
    itime = 0.0  # 演化的总虚时间
    Ebefore = 0.0
    for τ in τlis
        gates = get_gates(hams, τ; TrotterOrder=TrotterOrder)
        for it in 0:maxStep1τ
            @time "SU one step" errlis, prodNrm = simple_update_1step!(ipeps, Dk, gates; verbose=verbose, TrotterOrder=TrotterOrder)
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
            if it <= para[:minStep1τ]
                nothing
            elseif abs((E - Ebefore) / Ebefore) < para[:Etol] * τ^2
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
return a vector. [gateNNx, gateNNy], etc. \n 
"""
function get_gates(hams::Tuple, τ::Number; TrotterOrder=1)
    if length(hams) >= 3
        error("Only support nearest-neighbor interaction.")
    end
    gates = Vector{TensorMap}(undef, length(hams))
    if TrotterOrder == 1
        gates[1] = exp(-τ * hams[1])
        gates[2] = exp(-τ * hams[2])
    elseif TrotterOrder == 2
        gates[1] = exp(-τ * 0.5 * hams[1])
        gates[2] = exp(-τ * 0.5 * hams[2])
    else
        error("TrotterOrder should be 1 or 2")
    end
    return gates
end


function simple_update_1step!(ipeps::iPEPSΓΛ, Dk::Int, gates::Vector{TensorMap}; verbose=1, TrotterOrder=1)
    Lx = ipeps.Lx::Int
    Ly = ipeps.Ly::Int
    errlis = Vector{Float64}(undef, 2 * TrotterOrder * Lx * Ly)  # 总的 bond 数
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
        err, nrm = bond_proj_tb!(ipeps, xx, yy, Dk, gates[2])
        verbose > 1 ? println("纵向更新 [$xx, $yy], error=$err") : nothing
        errlis[Nb] = err
        prodNrm *= nrm
        Nb += 1
    end
    ## ======= 如果是2阶Trotter分解，反向再做一遍投影 ==============
    if TrotterOrder == 2
        for xx in Lx:-1:1, yy in Ly:-1:1
            err, nrm = bond_proj_tb!(ipeps, xx, yy, Dk, gates[2])
            verbose > 1 ? println("纵向更新 [$xx, $yy], error=$err") : nothing
            errlis[Nb] = err
            prodNrm *= nrm
            Nb += 1
        end
        for yy in Ly:-1:1, xx in Lx:-1:1
            err, nrm = bond_proj_lr!(ipeps, xx, yy, Dk, gates[1])
            verbose > 1 ? println("横向更新 [$xx, $yy], error=$err") : nothing
            errlis[Nb] = err
            prodNrm *= nrm
            Nb += 1
        end
    end
    return errlis, prodNrm
end


# function diag_proj_lu2rd!(ipeps::iPEPSΓΛ, xx::Int, yy::Int, Dk::Int, gateNNN::TensorMap)
#     errup1, errup2, nrmup1, nrmup2 = bond_proj_lu2rd_upPath!(ipeps, xx, yy, Dk, gateNNN)
#     errdn1, errdn2, nrmdn1, nrmdn2 = bond_proj_lu2rd_dnPath!(ipeps, xx, yy, Dk, gateNNN)
#     return errup1, errup2, errdn1, errdn2, nrmup1, nrmup2, nrmdn1, nrmdn2
# end


# function diag_proj_ru2ld!(ipeps::iPEPSΓΛ, xx::Int, yy::Int, Dk::Int, gateNNN::TensorMap)
#     errup1, errup2, nrmup1, nrmup2 = bond_proj_ru2ld_upPath!(ipeps, xx, yy, Dk, gateNNN)
#     errdn1, errdn2, nrmdn1, nrmdn2 = bond_proj_ru2ld_dnPath!(ipeps, xx, yy, Dk, gateNNN)
#     return errup1, errup2, errdn1, errdn2, nrmup1, nrmup2, nrmdn1, nrmdn2
# end
