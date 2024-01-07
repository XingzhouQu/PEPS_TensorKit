# ============================================================
"""
作用投影算符以裁剪环境, 左侧transfer张量. 注意要放入交换门！！
"""
function apply_proj_left!(ipeps::iPEPS, envs::iPEPSenv, projup::TensorMap, projdn::TensorMap, x::Int, y::Int)
    A = ipeps[x, y]
    Abar = bar(A)
    gate1 = swap_gate(space(A)[1], space(A)[2])
    gate2 = swap_gate(space(Abar)[4], space(A)[5])

    @tensor PTMMP[(tχNew, bχNew); (rupD, rdnD)] :=
        envs[x, y].transfer.l[tχin, bχin, rupDin, rdnDin] * projup[tχNew, tχin, tupDin, tdnDin] *
        gate1[rupDin, tdnDMin, rupDMin, tdnDin] * A[rupDMin, tupDin, p, rupD, upDMin] *
        Abar[rdnDin, tdnDMin, p, rdnDMin, dnDin] * gate2[rdnD, upDin, rdnDMin, upDMin] *
        projdn[bχin, upDin, dnDin, bχNew]

    envs[x, y].transfer.l = PTMMP
    return nothing
end


function apply_proj_right!(ipeps::iPEPS, envs::iPEPSenv, projup::TensorMap, projdn::TensorMap, x::Int, y::Int)
    A = ipeps[x, y]
    Abar = bar(A)
    gate1 = swap_gate(space(A)[1], space(A)[2])
    gate2 = swap_gate(space(A)[4], space(Abar)[5])   # 这里gate2的第四个要变成', 与环境初始化不一样--why??

    @tensor PTMMP[(lupD, ldnD, tχNew, bχNew); ()] :=
        envs[x, y].transfer.r[lupDin, ldnDin, tχin, bχin] * projdn[bχin, upDin, dnDin, bχNew] *
        gate2[ldnDMin, upDMin, ldnDin, upDin] * Abar[ldnD, tdnDMin, p, ldnDMin, dnDin] *
        A[lupDMin, tupDin, p, lupDin, upDMin] * gate1[lupD, tdnDMin, lupDMin, tdnDin] *
        projup[tχNew, tχin, tupDin, tdnDin]

    envs[x, y].transfer.r = PTMMP
    return nothing
end


function apply_proj_top!(ipeps::iPEPS, envs::iPEPSenv, projleft::TensorMap, projright::TensorMap, x::Int, y::Int)
    A = ipeps[x, y]
    Abar = bar(A)
    gate1 = swap_gate(space(A)[1], space(A)[2])
    gate2 = swap_gate(space(Abar)[4], space(A)[5])

    @tensor PTMMP[(lχNew, rχNew); (bupD, bdnD)] :=
        envs[x, y].transfer.t[lχin, rχin, bupDin, bdnDin] * projleft[lχin, lupDin, ldnDin, lχNew] *
        gate1[lupDin, bdnDMin, lupDMin, bdnDin] * A[lupDMin, bupDin, p, rupDin, bupDMin] *
        Abar[ldnDin, bdnDMin, p, rdnDMin, bdnD] * gate2[rdnDin, bupD, rdnDMin, bupDMin] *
        projright[rχNew, rχin, rupDin, rdnDin]

    envs[x, y].transfer.t = PTMMP
    return nothing
end


function apply_proj_bottom!(ipeps::iPEPS, envs::iPEPSenv, projleft::TensorMap, projright::TensorMap, x::Int, y::Int)
    A = ipeps[x, y]
    Abar = bar(A)
    gate1 = swap_gate(space(Abar)[1], space(Abar)[2])
    gate2 = swap_gate(space(Abar)[4], space(Abar)[5])

    @tensor PTMMP[(lχNew, tupD, tdnD, rχNew); ()] :=
        envs[x, y].transfer.b[lχin, tupDin, tdnDin, rχin] * projleft[lχin, lupDin, ldnDin, lχNew] *
        Abar[ldnDin, tdnDMin, p, rdnDMin, tdnDin] * gate1[lupDMin, tdnD, lupDin, tdnDMin] * 
        A[lupDMin, tupD, p, rupDin, tupDMin] * gate2[rdnDin, tupDMin, rdnDMin, tupDin] *
        projright[rχNew, rχin, rupDin, rdnDin]

    envs[x, y].transfer.b = PTMMP
    return nothing
end


# ============================================================
function apply_proj_ltCorner_updateL!(envs::iPEPSenv, proj::TensorMap, x::Int, y::Int)
    @tensor CPT[(rχNew, bχNew); ()] := (envs[x, y].corner.lt[rχin, bχin] * proj[bχin, upDin, dnDin, bχNew]) *
                                       envs[x, y].transfer.t[rχin, rχNew, upDin, dnDin]
    envs[x, y].corner.lt = CPT
    return nothing
end

function apply_proj_ltCorner_updateT!(envs::iPEPSenv, proj::TensorMap, x::Int, y::Int)
    @tensor CPT[(rχNew, bχNew); ()] := (envs[x, y].corner.lt[rχin, bχin] * proj[rχNew, rχin, upDin, dnDin]) *
                                       envs[x, y].transfer.l[bχin, bχNew, upDin, dnDin]
    envs[x, y].corner.lt = CPT
    return nothing
end

function apply_proj_lbCorner_updateL!(envs::iPEPSenv, proj::TensorMap, x::Int, y::Int)
    @tensor CPT[(tχNew, rχNew); ()] := (envs[x, y].corner.lb[tχin, rχin] * proj[tχNew, tχin, upDin, dnDin]) *
                                       envs[x, y].transfer.b[rχin, upDin, dnDin, rχNew]
    envs[x, y].corner.lb = CPT
    return nothing
end

function apply_proj_lbCorner_updateB!(envs::iPEPSenv, proj::TensorMap, x::Int, y::Int)
    @tensor CPT[(tχNew, rχNew); ()] := (envs[x, y].corner.lb[tχin, rχin] * proj[rχNew, rχin, upDin, dnDin]) *
                                       envs[x, y].transfer.l[tχNew, tχin, upDin, dnDin]
    envs[x, y].corner.lb = CPT
    return nothing
end

function apply_proj_rtCorner_updateR!(envs::iPEPSenv, proj::TensorMap, x::Int, y::Int)
    @tensor CPT[(lχNew, bχNew); ()] := (envs[x, y].corner.rt[lχin, bχin] * proj[bχin, upDin, dnDin, bχNew]) *
                                       envs[x, y].transfer.t[lχNew, lχin, upDin, dnDin]
    envs[x, y].corner.rt = CPT
    return nothing
end

function apply_proj_rtCorner_updateT!(envs::iPEPSenv, proj::TensorMap, x::Int, y::Int)
    @tensor CPT[(lχNew, bχNew); ()] := (envs[x, y].corner.rt[lχin, bχin] * proj[lχin, upDin, dnDin, lχNew]) *
                                       envs[x, y].transfer.r[upDin, dnDin, bχin, bχNew]
    envs[x, y].corner.rt = CPT
    return nothing
end

function apply_proj_rbCorner_updateR!(envs::iPEPSenv, proj::TensorMap, x::Int, y::Int)
    @tensor CPT[(lχNew, tχNew); ()] := (envs[x, y].corner.rb[lχin, tχin] * proj[tχNew, tχin, upDin, dnDin]) *
                                       envs[x, y].transfer.b[lχNew, upDin, dnDin, lχin]
    envs[x, y].corner.rb = CPT
    return nothing
end

function apply_proj_rbCorner_updateB!(envs::iPEPSenv, proj::TensorMap, x::Int, y::Int)
    @tensor CPT[(lχNew, tχNew); ()] := (envs[x, y].corner.rb[lχin, tχin] * proj[lχin, upDin, dnDin, lχNew]) *
                                       envs[x, y].transfer.r[upDin, dnDin, tχNew, tχin]
    envs[x, y].corner.rb = CPT
    return nothing
end