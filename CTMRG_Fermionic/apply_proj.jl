# TODO 把 ipepsbar 预先全都算出来，可以多占一点内存，省去循环中来来回回求 Abar 的时间 ===========
"""
作用投影算符以裁剪环境, 左侧transfer张量.
"""
function apply_proj_left!(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, projup::TensorMap, projdn::TensorMap, x::Int, y::Int)
    A = ipeps[x, y]
    Abar = ipepsbar[x, y]
    gate1 = swap_gate(space(A)[1], space(A)[2]; Eltype=eltype(A))
    gate2 = swap_gate(space(Abar)[4], space(A)[5]; Eltype=eltype(A))

    @tensor PTMMP[(tχNew, bχNew); (rupD, rdnD)] :=
        projup[tχNew, tχin, tupDin, tdnDin] * gate1[rupDin, tdnDMin, rupDMin, tdnDin] *
        A[rupDMin, tupDin, p, rupD, upDMin] * envs[x, y].transfer.l[tχin, bχin, rupDin, rdnDin] *
        Abar[rdnDin, tdnDMin, p, rdnDMin, dnDMin] * gate2[rdnD, upDin, rdnDMin, upDMin] *
        projdn[bχin, upDin, dnDin, bχNew]

    # 更新右侧的环境
    envs[x+1, y].transfer.l = PTMMP / norm(PTMMP, Inf)
    return nothing
end


function apply_proj_right!(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, projup::TensorMap, projdn::TensorMap, x::Int, y::Int)
    A = ipeps[x, y]
    Abar = ipepsbar[x, y]
    gate1 = swap_gate(space(A)[1], space(A)[2]; Eltype=eltype(A))
    gate2 = swap_gate(space(A)[4], space(Abar)[5]; Eltype=eltype(A))

    @tensor PTMMP[(lupD, ldnD, tχNew, bχNew); ()] :=
        projup[tχnew, tχin, tupDin, tdnDin] * gate1[lupD, tdnDMin, lupDMin, tdnDin] *
        A[lupDMin, tupDin, p, lupDin, upDMin] * envs[x, y].transfer.r[lupDin, ldnDin, tχin, bχin] *
        gate2[ldnDMin, upDMin, ldnDin, upDin] * Abar[ldnD, tdnDMin, p, ldnDMin, dnDin] *
        projdn[bχin, upDin, dnDin, bχnew]

    # 更新左侧的环境
    envs[x-1, y].transfer.r = PTMMP / norm(PTMMP, Inf)
    return nothing
end


function apply_proj_top!(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, projleft::TensorMap, projright::TensorMap, x::Int, y::Int)
    A = ipeps[x, y]
    Abar = ipepsbar[x, y]
    gate1 = swap_gate(space(A)[1], space(A)[2]; Eltype=eltype(A))
    gate2 = swap_gate(space(Abar)[4], space(A)[5]; Eltype=eltype(A))

    @tensor PTMMP[(lχNew, rχNew); (bupD, bdnD)] :=
        projleft[lχin, lupDin, ldnDin, lχNew] * gate1[lupDin, bdnDMin, lupDMin, bdnDin] *
        A[lupDMin, bupDin, p, rupDin, bupDMin] * envs[x, y].transfer.t[lχin, rχin, bupDin, bdnDin] *
        Abar[ldnDin, bdnDMin, p, rdnDMin, bdnD] * gate2[rdnDin, bupD, rdnDMin, bupDMin] *
        projright[rχNew, rχin, rupDin, rdnDin]

    # 更新下侧的环境
    envs[x, y+1].transfer.t = PTMMP / norm(PTMMP, Inf)
    return nothing
end


function apply_proj_bottom!(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, projleft::TensorMap, projright::TensorMap, x::Int, y::Int)
    A = ipeps[x, y]
    Abar = ipepsbar[x, y]
    gate1 = swap_gate(space(Abar)[1], space(Abar)[2]; Eltype=eltype(A))
    gate2 = swap_gate(space(Abar)[4], space(Abar)[5]; Eltype=eltype(A))

    @tensor PTMMP[(lχNew, tupD, tdnD, rχNew); ()] :=
        projleft[lχin, lupDin, ldnDin, lχNew] * gate1[lupDMin, tdnD, lupDin, tdnDMin] *
        Abar[ldnDin, tdnDMin, p, rdnDMin, tdnDin] * envs[x, y].transfer.b[lχin, tupDin, tdnDin, rχin] *
        A[lupDMin, tupD, p, rupDin, tupDMin] * gate2[rdnDin, tupDMin, rdnDMin, tupDin] *
        projright[rχNew, rχin, rupDin, rdnDin]

    # 更新上侧的环境
    envs[x, y-1].transfer.b = PTMMP / norm(PTMMP, Inf)
    return nothing
end


# ============================================================
function apply_proj_ltCorner_updateL!(envs::iPEPSenv, proj::TensorMap, x::Int, y::Int)
    @tensor CPT[(rχNew, bχNew); ()] := (envs[x, y].corner.lt[rχin, bχin] * proj[bχin, upDin, dnDin, bχNew]) *
                                       envs[x, y].transfer.t[rχin, rχNew, upDin, dnDin]
    envs[x+1, y].corner.lt = CPT / norm(CPT, Inf)
    return nothing
end

function apply_proj_ltCorner_updateT!(envs::iPEPSenv, proj::TensorMap, x::Int, y::Int)
    @tensor CPT[(rχNew, bχNew); ()] := (envs[x, y].corner.lt[rχin, bχin] * proj[rχNew, rχin, upDin, dnDin]) *
                                       envs[x, y].transfer.l[bχin, bχNew, upDin, dnDin]
    envs[x, y+1].corner.lt = CPT / norm(CPT, Inf)
    return nothing
end

function apply_proj_lbCorner_updateL!(envs::iPEPSenv, proj::TensorMap, x::Int, y::Int)
    @tensor CPT[(tχNew, rχNew); ()] := (envs[x, y].corner.lb[tχin, rχin] * proj[tχNew, tχin, upDin, dnDin]) *
                                       envs[x, y].transfer.b[rχin, upDin, dnDin, rχNew]
    envs[x+1, y].corner.lb = CPT / norm(CPT, Inf)
    return nothing
end

function apply_proj_lbCorner_updateB!(envs::iPEPSenv, proj::TensorMap, x::Int, y::Int)
    @tensor CPT[(tχNew, rχNew); ()] := (envs[x, y].corner.lb[tχin, rχin] * proj[rχNew, rχin, upDin, dnDin]) *
                                       envs[x, y].transfer.l[tχNew, tχin, upDin, dnDin]
    envs[x, y-1].corner.lb = CPT / norm(CPT, Inf)
    return nothing
end

function apply_proj_rtCorner_updateR!(envs::iPEPSenv, proj::TensorMap, x::Int, y::Int)
    @tensor CPT[(lχNew, bχNew); ()] := (envs[x, y].corner.rt[lχin, bχin] * proj[bχin, upDin, dnDin, bχNew]) *
                                       envs[x, y].transfer.t[lχNew, lχin, upDin, dnDin]
    envs[x-1, y].corner.rt = CPT / norm(CPT, Inf)
    return nothing
end

function apply_proj_rtCorner_updateT!(envs::iPEPSenv, proj::TensorMap, x::Int, y::Int)
    @tensor CPT[(lχNew, bχNew); ()] := (envs[x, y].corner.rt[lχin, bχin] * proj[lχin, upDin, dnDin, lχNew]) *
                                       envs[x, y].transfer.r[upDin, dnDin, bχin, bχNew]
    envs[x, y+1].corner.rt = CPT / norm(CPT, Inf)
    return nothing
end

function apply_proj_rbCorner_updateR!(envs::iPEPSenv, proj::TensorMap, x::Int, y::Int)
    @tensor CPT[(lχNew, tχNew); ()] := (envs[x, y].corner.rb[lχin, tχin] * proj[tχNew, tχin, upDin, dnDin]) *
                                       envs[x, y].transfer.b[lχNew, upDin, dnDin, lχin]
    envs[x-1, y].corner.rb = CPT / norm(CPT, Inf)
    return nothing
end

function apply_proj_rbCorner_updateB!(envs::iPEPSenv, proj::TensorMap, x::Int, y::Int)
    @tensor CPT[(lχNew, tχNew); ()] := (envs[x, y].corner.rb[lχin, tχin] * proj[lχin, upDin, dnDin, lχNew]) *
                                       envs[x, y].transfer.r[upDin, dnDin, tχNew, tχin]
    envs[x, y-1].corner.rb = CPT / norm(CPT, Inf)
    return nothing
end