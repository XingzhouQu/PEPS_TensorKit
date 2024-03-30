# ============================================================
"""
作用投影算符以裁剪环境, 左侧transfer张量.
"""
function apply_proj_left!(ipeps::iPEPS, envs::iPEPSenv, projup::TensorMap, projdn::TensorMap, x::Int, y::Int)
    A = ipeps[x, y]
    Abar = A'

    @tensor PTMMP[(tχNew, bχNew); (rupD, rdnD)] :=
        envs[x, y].transfer.l[tχin, bχin, rupDin, rdnDin] * projup[tχNew, tχin, tupDin, tdnDin] *
        A[rupDin, tupDin, p, rupD, upDin] * Abar[rdnD, dnDin, rdnDin, tdnDin, p] *
        projdn[bχin, upDin, dnDin, bχNew]

    # 更新右侧的环境
    envs[x+1, y].transfer.l = PTMMP
    return nothing
end


function apply_proj_right!(ipeps::iPEPS, envs::iPEPSenv, projup::TensorMap, projdn::TensorMap, x::Int, y::Int)
    A = ipeps[x, y]
    Abar = A'

    @tensor PTMMP[(lupD, ldnD, tχNew, bχNew); ()] :=
        envs[x, y].transfer.r[lupDin, ldnDin, tχin, bχin] * projup[tχNew, tχin, tupDin, tdnDin] *
        A[lupD, tupDin, p, lupDin, upDin] * Abar[ldnDin, dnDin, ldnD, tdnDin, p] *
        projdn[bχin, upDin, dnDin, bχNew]

    # 更新左侧的环境
    envs[x-1, y].transfer.r = PTMMP
    return nothing
end


function apply_proj_top!(ipeps::iPEPS, envs::iPEPSenv, projleft::TensorMap, projright::TensorMap, x::Int, y::Int)
    A = ipeps[x, y]
    Abar = A'

    @tensor PTMMP[(lχNew, rχNew); (bupD, bdnD)] :=
        envs[x, y].transfer.t[lχin, rχin, bupDin, bdnDin] * projleft[lχin, lupDin, ldnDin, lχNew] *
        A[lupDin, bupDin, p, rupDin, bupD] * Abar[rdnDin, bdnD, ldnDin, bdnDin, p] *
        projright[rχNew, rχin, rupDin, rdnDin]

    # 更新下侧的环境
    envs[x, y+1].transfer.t = PTMMP
    return nothing
end


function apply_proj_bottom!(ipeps::iPEPS, envs::iPEPSenv, projleft::TensorMap, projright::TensorMap, x::Int, y::Int)
    A = ipeps[x, y]
    Abar = A'

    @tensor PTMMP[(lχNew, tupD, tdnD, rχNew); ()] :=
        envs[x, y].transfer.b[lχin, tupDin, tdnDin, rχin] * projleft[lχin, lupDin, ldnDin, lχNew] *
        Abar[rdnDin, tdnDin, ldnDin, tdnD, p] * A[lupDin, tupD, p, rupDin, tupDin] *
        projright[rχNew, rχin, rupDin, rdnDin]

    # 更新上侧的环境
    envs[x, y-1].transfer.b = PTMMP
    return nothing
end


# ============================================================
function apply_proj_ltCorner_updateL!(envs::iPEPSenv, proj::TensorMap, x::Int, y::Int)
    @tensor CPT[(rχNew, bχNew); ()] := (envs[x, y].corner.lt[rχin, bχin] * proj[bχin, upDin, dnDin, bχNew]) *
                                       envs[x, y].transfer.t[rχin, rχNew, upDin, dnDin]
    envs[x+1, y].corner.lt = CPT
    return nothing
end

function apply_proj_ltCorner_updateT!(envs::iPEPSenv, proj::TensorMap, x::Int, y::Int)
    @tensor CPT[(rχNew, bχNew); ()] := (envs[x, y].corner.lt[rχin, bχin] * proj[rχNew, rχin, upDin, dnDin]) *
                                       envs[x, y].transfer.l[bχin, bχNew, upDin, dnDin]
    envs[x, y+1].corner.lt = CPT
    return nothing
end

function apply_proj_lbCorner_updateL!(envs::iPEPSenv, proj::TensorMap, x::Int, y::Int)
    @tensor CPT[(tχNew, rχNew); ()] := (envs[x, y].corner.lb[tχin, rχin] * proj[tχNew, tχin, upDin, dnDin]) *
                                       envs[x, y].transfer.b[rχin, upDin, dnDin, rχNew]
    envs[x+1, y].corner.lb = CPT
    return nothing
end

function apply_proj_lbCorner_updateB!(envs::iPEPSenv, proj::TensorMap, x::Int, y::Int)
    @tensor CPT[(tχNew, rχNew); ()] := (envs[x, y].corner.lb[tχin, rχin] * proj[rχNew, rχin, upDin, dnDin]) *
                                       envs[x, y].transfer.l[tχNew, tχin, upDin, dnDin]
    envs[x, y-1].corner.lb = CPT
    return nothing
end

function apply_proj_rtCorner_updateR!(envs::iPEPSenv, proj::TensorMap, x::Int, y::Int)
    @tensor CPT[(lχNew, bχNew); ()] := (envs[x, y].corner.rt[lχin, bχin] * proj[bχin, upDin, dnDin, bχNew]) *
                                       envs[x, y].transfer.t[lχNew, lχin, upDin, dnDin]
    envs[x-1, y].corner.rt = CPT
    return nothing
end

function apply_proj_rtCorner_updateT!(envs::iPEPSenv, proj::TensorMap, x::Int, y::Int)
    @tensor CPT[(lχNew, bχNew); ()] := (envs[x, y].corner.rt[lχin, bχin] * proj[lχin, upDin, dnDin, lχNew]) *
                                       envs[x, y].transfer.r[upDin, dnDin, bχin, bχNew]
    envs[x, y+1].corner.rt = CPT
    return nothing
end

function apply_proj_rbCorner_updateR!(envs::iPEPSenv, proj::TensorMap, x::Int, y::Int)
    @tensor CPT[(lχNew, tχNew); ()] := (envs[x, y].corner.rb[lχin, tχin] * proj[tχNew, tχin, upDin, dnDin]) *
                                       envs[x, y].transfer.b[lχNew, upDin, dnDin, lχin]
    envs[x-1, y].corner.rb = CPT
    return nothing
end

function apply_proj_rbCorner_updateB!(envs::iPEPSenv, proj::TensorMap, x::Int, y::Int)
    @tensor CPT[(lχNew, tχNew); ()] := (envs[x, y].corner.rb[lχin, tχin] * proj[lχin, upDin, dnDin, lχNew]) *
                                       envs[x, y].transfer.r[upDin, dnDin, tχNew, tχin]
    envs[x, y-1].corner.rb = CPT
    return nothing
end