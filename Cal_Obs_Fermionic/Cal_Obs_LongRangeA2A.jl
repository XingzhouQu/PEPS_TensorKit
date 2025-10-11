struct pre_compute
    Rx::Int
    Ry::Int
    # 计算横向两点关联要用到的中间环境 h_env[x, y, z] 表示点 [x, y] 出发，宽度为 z 的横向关联环境
    h_env::AbstractArray{AbstractTensorMap,3}
    # 计算纵向两点关联要用到的中间环境 v_env[x, y, z] 表示点 [x, y] 出发，宽度为 z 的纵向关联环境
    v_env::AbstractArray{AbstractTensorMap,3}
    function pre_compute(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, Rx::Int, Ry::Int)
        Lx = ipeps.Lx
        Ly = ipeps.Ly
        h_env = Array{AbstractTensorMap}(undef, Lx, Ly, Rx)
        v_env = Array{AbstractTensorMap}(undef, Lx, Ly, Ry)
        _pre_compute_h_env!(ipeps, ipepsbar, envs, h_env)
        _pre_compute_v_env!(ipeps, ipepsbar, envs, v_env)

        return new(Rx, Ry, h_env, v_env)
    end
end

# 上下封口，用于计算 y1=y2, x1≠x2 的关联
function _pre_compute_v_env!(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, v_env::AbstractArray{AbstractTensorMap,3})
    (Lx, Ly, Ry) = size(v_env)
    # 遍历所有点
    @floop for val in CartesianIndices((Lx, Ly))
        (xx, yy) = Tuple(val)
        # 收缩上方环境
        CapUp = _get_toppart(ipeps, ipepsbar, envs, xx, yy)
        # 逐一收缩下方环境
        for extd in 1:Ry
            tmp = _shrink_env_tb(envs, xx, yy, CapUp, extd)
            v_env[xx, yy, extd] = tmp
            if extd < Ry
                CapUp = _update_CapUp(CapUp, ipeps, ipepsbar, xx, yy, extd)
            end
        end
    end
    return nothing
end

# 左右封口，用于计算 x1=x2, y1≠y2 的关联
function _pre_compute_h_env!(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, h_env::AbstractArray{AbstractTensorMap,3})
    (Lx, Ly, Rx) = size(h_env)
    # 遍历所有点
    @floop for val in CartesianIndices((Lx, Ly))
        (xx, yy) = Tuple(val)
        # 收缩左部环境
        CapL = _get_leftpart(ipeps, ipepsbar, envs, xx, yy)
        # 逐一收缩右部环境
        for extd in 1:Rx
            tmp = _shrink_env_lr(envs, xx, yy, CapL, extd)
            h_env[xx, yy, extd] = tmp
            if extd < Rx
                CapL = _update_CapL(CapL, ipeps, ipepsbar, xx, yy, extd)
            end
        end
    end
    return nothing
end


# 计算上方环境
function _get_toppart(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, xx::Int, yy::Int)
    M = ipeps[xx, yy]::TensorMap
    Mbar = ipepsbar[xx, yy]::TensorMap
    gate1 = swap_gate(space(M)[1], space(Mbar)[2]; Eltype=eltype(M))
    gate2 = swap_gate(space(M)[5], space(Mbar)[4]; Eltype=eltype(M))

    @tensoropt toppart[lχ, rχ; lupD, ldnD, rupD, rdnD, bupD, bdnD] :=
        M[lupDMin, bupDin, p, rupD, bupDMin] * Mbar[ldnD, bdnDMin, p, ldnDMin, bdnD] *
        gate2[bupD, rdnD, bupDMin, ldnDMin] * gate1[lupD, bdnDin, lupDMin, bdnDMin] *
        envs[xx, yy].transfer.t[lχ, rχ, bupDin, bdnDin]

    return toppart
end


# 计算左侧环境
function _get_leftpart(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, xx::Int, yy::Int)
    M = ipeps[x, y]::TensorMap
    Mbar = ipepsbar[x, y]::TensorMap
    gate1 = swap_gate(space(M)[1], space(Mbar)[2]; Eltype=eltype(M))
    gate2 = swap_gate(space(M)[5], space(Mbar)[4]; Eltype=eltype(M))

    @tensoropt leftpart[tχ, bχ; tupD, tdnD, bupD, bdnD, rupD, rdnD] :=
        M[rupDMin, tupD, p, rupD, tupDMin] * Mbar[rdnDin, tdnDMin, p, rdnDMin, bdnD] *
        gate2[bupD, rdnD, tupDMin, rdnDMin] * gate1[rupDin, tdnD, rupDMin, tdnDMin] *
        envs[xx, yy].transfer.l[tχ, bχ, rupDin, rdnDin]

    return leftpart
end

# ncon convention: Open indices are negative (order -1, -2, ...), contracted indices are positive (contraction order 1, 2, ...)
# 给定上环境 CapUp, 将其下方的环境收缩进来，形成完整环境用于后续计算
function _shrink_env_tb(envs::iPEPSenv, xx::Int, yy::Int, CapUp::AbstractTensorMap, extd::Int)
    CapDn = envs[xx, yy+extd-1].transfer.b
    rk = numind(CapUp)
    idxA = [-i for i in 3:rk]  # 开放指标
    append!(idxA, [1, 2])  # 缩并掉最后两个指标

    # 返回张量的指标顺序：[下左χ, 下右χ， 上左χ， 上右χ; 其他D]
    return ncon((CapUp, CapDn), (idxA, [-1, 1, 2, -2]))
end

# 给定左环境 CapL, 将其右侧的环境收缩进来，形成完整环境用于后续计算
function _shrink_env_lr(envs::iPEPSenv, xx::Int, yy::Int, CapL::AbstractTensorMap, extd::Int)
    CapR = envs[xx+extd-1, yy].transfer.r
    rk = numind(CapL)
    idxA = [-i for i in 3:rk]  # 开放指标
    append!(idxA, [1, 2])  # 缩并掉最后两个指标

    # 返回张量的指标顺序：[右上χ, 右下χ， 左上χ， 左下χ; 其他D]
    return ncon((CapL, CapR), (idxA, [1, 2, -1, -2]))
end

# 给定上环境 CapUp, 将其下方的两个 iPEPS 张量收缩进来
function _update_CapUp(CapUp::AbstractTensorMap, ipeps::iPEPS, ipepsbar::iPEPS, xx::Int, yy::Int, extd::Int)
    M = ipeps[xx, yy+extd]::TensorMap
    Mbar = ipepsbar[xx, yy+extd]::TensorMap
    gate1 = swap_gate(space(M)[1], space(Mbar)[2]; Eltype=eltype(M))
    gate2 = swap_gate(space(M)[5], space(Mbar)[4]; Eltype=eltype(M))

    rk = numind(CapUp)
    idxA = [-i for i in 1:rk-2]  # 开放指标
    append!(idxA, [6, 7])  # 缩并掉最后两个指标

    rslt = ncon((CapUp, M, Mbar, gate1, gate2), (idxA, [2, 6, 1, -rk - 1, 4], [-rk, 3, 1, 5, -rk - 4], [-rk + 1, 7, 2, 3], [-rk - 3, -rk - 2, 4, 5]))

    return rslt / norm(rslt, Inf)  # 归一化
end

# 给定左环境 CapL, 将其右侧的两个 iPEPS 张量收缩进来
function _update_CapL(CapL::AbstractTensorMap, ipeps::iPEPS, ipepsbar::iPEPS, xx::Int, yy::Int, extd::Int)
    M = ipeps[xx+extd, yy]::TensorMap
    Mbar = ipepsbar[xx+extd, yy]::TensorMap
    gate1 = swap_gate(space(M)[1], space(Mbar)[2]; Eltype=eltype(M))
    gate2 = swap_gate(space(M)[5], space(Mbar)[4]; Eltype=eltype(M))

    rk = numind(CapL)
    idxA = [-i for i in 1:rk-2]  # 开放指标
    append!(idxA, [6, 7])  # 缩并掉最后两个指标

    rslt = ncon((CapL, M, Mbar, gate1, gate2), (idxA, [2, -rk + 1, 1, -rk - 3, 4], [7, 3, 1, 5, -rk - 2], [6, -rk, 2, 3], [-rk - 1, -rk - 4, 4, 5]))

    return rslt / norm(rslt, Inf)  # 归一化
end


"""
计算两点长程关联。 \n
返回一系列关联函数矩阵，每个矩阵的维度为 (Rx, Ry)，其中 R 为实空间延伸的格点数。即 [1:Rx, 1:Ry] 内的 all-to-all 关联。 \n 
此函数将观测量 O 分为 OpL, OpR, 分别预先与环境收缩。计算多组算符观测量时效率不如 Cal_Obs_2site \n 
"""
function Cal_Obs_2site_long_range(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, Gates::Vector{String}, para::Dict{Symbol,Any}, Rx::Int, Ry::Int, get_op::Function)
    Lx = ipeps.Lx
    Ly = ipeps.Ly
    pre_compute_env = pre_compute(ipeps, ipepsbar, envs, Rx, Ry)
    GC.gc()
    rslt = Dict{String,Matrix}()
    for gate in Gates
        rslt[gate] = Matrix{Number}(undef, 0, 5)
    end
    # 循环计算所有关联
    for (ind1, val1) in enumerate(CartesianIndices((Rx, Ry)))
        (x1, y1) = Tuple(val1)  # 作为起始点
        # onsite terms. # Obs_onsite::Dict{String,Number}
        Obs_onsite = Cal_Obs_onsite(ipeps, ipepsbar, envs, Gates, para, get_op; site=[x1, y1])
        # keep x, scan y. # Obs_h2site::Dict{String, Matrix{Number}(5列)}
        Obs_h2site = Cal_Obs_h(ipeps, ipepsbar, envs, pre_compute_env, Gates, para, get_op, Ry; site=[x1, y1])
        # keep y, scan x. # Obs_v2site::Dict{String, Matrix{Number}(5列)}
        Obs_v2site = Cal_Obs_v(ipeps, ipepsbar, envs, pre_compute_env, Gates, para, get_op, Rx; site=[x1, y1])
        # scan others. # Obs_diag2site::Dict{String, Matrix{Number}(5列)}
        Obs_diag2site = Cal_Obs_diag(ipeps, ipepsbar, envs, pre_compute_env, Gates, para, get_op; site=[x1, y1, x1 + 1, y1 + 1])

        for gate in Gates
            rslt[gate] = vcat(rslt[gate], [x1, y1, x1, y1, Obs_onsite[gate]])
            rslt[gate] = vcat(rslt[gate], Obs_h2site[gate])
            rslt[gate] = vcat(rslt[gate], Obs_v2site[gate])
            rslt[gate] = vcat(rslt[gate], Obs_diag2site[gate])
        end
        GC.gc()
    end

    for gate in Gates
        rslt[gate] = sortslices(rslt[gate], dims=1)
    end
    return rslt
end


# x1 == x2
function Cal_Obs_h(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, pre_compute_env::pre_compute, Gates::Vector{String}, para::Dict{Symbol,Any}, get_op::Function, Ry::Int; site=[1, 1])
    (x1, y1) = site
    Obs_h2site = Dict{String,Matrix}()
    for gate in Gates
        Obs_h2site[gate] = Matrix{Number}(undef, 0, 5)
    end
    # 起始张量
    M1 = ipeps[x1, y1]
    M1bar = ipepsbar[x1, y1]
    swgatet1 = swap_gate(space(M1)[1], space(M1bar)[2]; Eltype=eltype(M1))
    swgatet2 = swap_gate(space(M1bar)[3], space(M1bar)[4]; Eltype=eltype(M1))
    swgatet3 = swap_gate(space(M1)[3], space(swgatet2)[2]; Eltype=eltype(M1))
    swgatet4 = swap_gate(space(M1)[5], space(swgatet3)[2]; Eltype=eltype(M1))
    @tensor opt = true toppart[pup1, Dupin, l12l2; pdn1, Ddnin, r12r2] :=
        envs[x1, y1].corner.lt[lt2t, lt2l1] * envs[x1, y1].transfer.l[lt2l1, l12l2, l12Dup, l12Ddn] *
        envs[x1, y1].transfer.t[lt2t, rt2t, t2Dup, t2Ddn] * envs[x1, y1].corner.rt[rt2t, rt2r1] *
        swgatet1[l12Dup, t2Ddn, l12Dupin, t2Ddnin] * M1[l12Dupin, t2Dup, pup1in, r12Dup, Dupin2] *
        M1bar[l12Ddn, t2Ddnin, pdn1in, r12Ddnin, Ddnin] * swgatet2[pdn1, r12Ddnin2, pdn1in, r12Ddnin] *
        swgatet3[pup1, r12Ddnin3, pup1in, r12Ddnin2] * swgatet4[Dupin, r12Ddn, Dupin2, r12Ddnin3] *
        envs[x1, y1].transfer.r[r12Dup, r12Ddn, rt2r1, r12r2]

    midpart = nothing
    for deltaY in 1:Ry # 顺序执行，不要并行
        # 目标张量
        M2 = ipeps[x1, y1+deltaY]
        M2bar = ipepsbar[x1, y1+deltaY]
        swgateb4 = swap_gate(space(M2bar)[4], space(M2)[5]; Eltype=eltype(M1))
        swgateb3 = swap_gate(space(M2)[1], space(M2)[3]; Eltype=eltype(M1))
        swgateb2 = swap_gate(space(M2bar)[3], space(swgateb3)[1]; Eltype=eltype(M1))
        swgateb1 = swap_gate(space(M2bar)[2], space(swgateb2)[2]; Eltype=eltype(M1))
        @tensor opt = true bottompart[pup2, Dupin, l12l2; pdn2, Ddnin, r12r2] :=
            envs[x2, y2].transfer.l[l12l2, lb2l2, l22Dup, l22Ddn] *
            envs[x2, y2].corner.lb[lb2l2, lb2b] * swgateb1[Ddnin, l22Dup, Ddninin, l22Dupin3] *
            M2bar[l22Ddn, Ddninin, pdn2in, r22Ddnin, b2Ddn] * swgateb2[pdn2, l22Dupin3, pdn2in, l22Dupin2] *
            swgateb3[l22Dupin2, pup2, l22Dupin, pup2in] * M2[l22Dupin, Dupin, pup2in, r22Dup, b2Dupin] *
            swgateb4[r22Ddn, b2Dup, r22Ddnin, b2Dupin] * envs[x2, y2].transfer.r[r22Dup, r22Ddn, r12r2, rb2r2] *
            envs[x2, y2].transfer.b[lb2b, b2Dup, b2Ddn, rb2b] * envs[x2, y2].corner.rb[rb2b, rb2r2]

        if deltaY == 1  # 不需要中间部分
            @tensor opt = true ψ□ψ[pup1, pup2; pdn1, pdn2] := toppart[pup1, Dupin, l12l2; pdn1, Ddnin, r12r2] * bottompart[pup2, Dupin, l12l2; pdn2, Ddnin, r12r2]
        else  # 引入中间部分
            update_midpart_h!(midpart, pre_compute_env, x1, y1, deltaY)
            @tensor opt = true ψ□ψ[pup1, pup2; pdn1, pdn2] :=
                toppart[pup1, Dupin, l12l2; pdn1, Ddnin, r12r2] * midpart[r12r2, r12r2p, l12l2, l12l2p, Dupin, Ddnin, Dupin2, Ddnin2] *
                bottompart[pup2, Dupin2, l12l2p; pdn2, Ddnin2, r12r2pl2p; pdn2, Ddnin, r12r2p]
        end
        @tensor nrm = ψ□ψ[p1, p2, p1, p2]

        # 对这两点循环所有需要计算的观测量，记录得到的值
        for gate in Gates
            OpL, OpR = get_op(gate, para; Nop=2)
            # get onsite Operator
            rk = numind(OpL)
            if rk == 3
                @tensor OnsiteOp[p1, p3; p2, p4] = OpL[p1, p2, a] * OpR[a, p3, p4]
            elseif rk == 2
                @tensor OnsiteOp[p1, p3; p2, p4] = OpL[p1, p2] * OpR[p3, p4]
            else
                error("rank of OpL and OpR must be 2 or 3")
            end
            @tensor tmp = ψ□ψ[pup1, pup2; pdn1, pdn2] * OpL[pdn1, pup1, a] * OpR[a, pdn2, pup2]
            Obs_h2site[gate] = vcat(Obs_h2site[gate], [x1, y1, x1, y1 + deltaY, tmp / nrm])
        end
    end
    return Obs_h2site
end

# 迭代调用此函数，利用 pre_compute_env[x, y, 1] 更新中间部分
function update_midpart_h!(midpart::Union{nothing,AbstractTensorMap}, pre_compute_env::pre_compute, x1::Int, y1::Int, deltaY::Int)
    if midpart === nothing
        midpart = pre_compute_env.h_env[x1, y1, 1]
    else
        @tensor midpart[rtχ, rbχ, ltχ, lbχ, Dup, Ddn, Dup2, Ddn2] =
            midpart[rtχ, rχin, ltχ, lχin, Dup, Ddn, Dupin, Ddnin] * pre_compute_env.h_env[x1, y1+deltaY-1, 1][rχin, rbχ, lχin, lbχ, Dupin, Ddnin, Dup2, Ddn2]
        midpart = midpart / norm(midpart, Inf)
    end
    return nothing
end

# y1 == y2
function Cal_Obs_v(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, pre_compute_env::pre_compute, Gates::Vector{String}, para::Dict{Symbol,Any}, get_op::Function, Rx::Int; site=[1, 1])
    (x1, y1) = site
    Obs_v2site = Dict{String,Matrix}()
    for gate in Gates
        Obs_v2site[gate] = Matrix{Number}(undef, 0, 5)
    end
    # 起始张量
    M1 = ipeps[x1, y1]
    M1bar = ipepsbar[x1, y1]
    swgatel1 = swap_gate(space(M1)[1], space(M1bar)[2]; Eltype=eltype(M1))
    swgatel2 = swap_gate(space(M1)[3], space(M1)[5]; Eltype=eltype(M1))
    swgatel3 = swap_gate(space(M1bar)[3], space(swgatel2)[2]; Eltype=eltype(M1))
    swgatel4 = swap_gate(space(swgatel3)[2], space(M1bar)[4]; Eltype=eltype(M1))
    @tensor opt = true leftpart[pup1, Dupin, t12t2; pdn1, Ddnin, b12b2] :=
        envs[x1, y1].corner.lt[lt2t1, lt2l] * envs[x1, y1].transfer.t[lt2t1, t12t2, t12Dup, t12Ddn] *
        swgatel1[l2Dup, t12Ddn, l2Dupin, t12Ddnin] * M1[l2Dupin, t12Dup, pup1in, Dupin, b12Dupin] *
        swgatel2[pup1, b12Dupin2, pup1in, b12Dupin] * swgatel3[pdn1, b12Dupin3, pdn1in, b12Dupin2] *
        M1bar[l2Ddn, t12Ddnin, pdn1in, Ddnin2, b12Ddn] * envs[x1, y1].transfer.l[lt2l, lb2l, l2Dup, l2Ddn] *
        swgatel4[b12Dup, Ddnin, b12Dupin3, Ddnin2] * envs[x1, y1].transfer.b[lb2b1, b12Dup, b12Ddn, b12b2] *
        envs[x1, y1].corner.lb[lb2l, lb2b1]

    midpart = nothing
    for deltaX in 1:Rx # 顺序执行，不要并行
        M2 = ipeps[x1+deltaX, y1]
        M2bar = ipepsbar[x1+deltaX, y1]
        swgater4 = swap_gate(space(M2)[5], space(M2bar)[4]; Eltype=eltype(M1))
        swgater3 = swap_gate(space(M2bar)[3], space(M2bar)[2]; Eltype=eltype(M1))
        swgater2 = swap_gate(space(M2)[3], space(swgater3)[2]; Eltype=eltype(M1))
        swgater1 = swap_gate(space(M2)[1], space(swgater2)[2]; Eltype=eltype(M1))
        @tensor opt = true rightpart[pup2, t12t2, Dupin; pdn2, Ddnin, b12b2] :=
            swgater1[Dupin, t22Ddn, Dupinin, t22Ddnin3] *
            envs[x2, y2].transfer.t[t12t2, rt2t2, t22Dup, t22Ddn] * M2[Dupinin, t22Dup, pup2in, r2Dup, b22Dupin] *
            swgater2[pup2, t22Ddnin3, pup2in, t22Ddnin2] * swgater3[pdn2, t22Ddnin2, pdn2in, t22Ddnin] *
            M2bar[Ddnin, t22Ddnin, pdn2in, r2Ddnin, b22Ddn] * envs[x2, y2].transfer.b[b12b2, b22Dup, b22Ddn, rb2b2] *
            envs[x2, y2].corner.rt[rt2t2, rt2r] * swgater4[b22Dup, r2Ddn, b22Dupin, r2Ddnin] *
            envs[x2, y2].transfer.r[r2Dup, r2Ddn, rt2r, rb2r] * envs[x2, y2].corner.rb[rb2b2, rb2r]

        if deltaX == 1  # 不需要中间部分
            @tensor opt = true ψ□ψ[pup1, pup2; pdn1, pdn2] := leftpart[pup1, Dupin, t12t2, pdn1, Ddnin, b12b2] * rightpart[pup2, t12t2, Dupin, pdn2, Ddnin, b12b2]
        else  # 引入中间部分
            update_midpart_v!(midpart, pre_compute_env, x1, y1, deltaX)
            @tensor opt = true ψ□ψ[pup1, pup2; pdn1, pdn2] :=
                leftpart[pup1, Dupin, t12t2, pdn1, Ddnin, b12b2] * midpart[b12b2, b12b2p, t12t2, t12t2p, Dupin, Ddnin, Dupin2, Ddnin2] *
                rightpart[pup2, t12t2p, Dupin2, pdn2, Ddnin2, b12b2p]
        end
        @tensor nrm = ψ□ψ[p1, p2, p1, p2]

        # 对这两点循环所有需要计算的观测量，记录得到的值
        for gate in Gates
            OpL, OpR = get_op(gate, para; Nop=2)
            # get onsite Operator
            rk = numind(OpL)
            if rk == 3
                @tensor OnsiteOp[p1, p3; p2, p4] = OpL[p1, p2, a] * OpR[a, p3, p4]
            elseif rk == 2
                @tensor OnsiteOp[p1, p3; p2, p4] = OpL[p1, p2] * OpR[p3, p4]
            else
                error("rank of OpL and OpR must be 2 or 3")
            end
            @tensor tmp = ψ□ψ[pup1, pup2; pdn1, pdn2] * OpL[pdn1, pup1, a] * OpR[a, pdn2, pup2]
            Obs_v2site[gate] = vcat(Obs_v2site[gate], [x1, y1, x1, y1 + deltaY, tmp / nrm])
        end

    end
    return Obs_v2site
end

# 迭代调用此函数，利用 pre_compute_env[x, y, 1] 更新中间部分
function update_midpart_v!(midpart::Union{nothing,AbstractTensorMap}, pre_compute_env::pre_compute, x1::Int, y1::Int, deltaX::Int)
    if midpart === nothing
        midpart = pre_compute_env.v_env[x1, y1, 1]
    else
        @tensor midpart[lbχ, rbχ, ltχ, rtχ, Dup, Ddn, Dup2, Ddn2] =
            midpart[lbχ, bχin, ltχ, tχin, Dup, Ddn, Dupin, Ddnin] * pre_compute_env.v_env[x1+deltaX-1, y1, 1][bχin, rbχ, tχin, rtχ, Dupin, Ddnin, Dup2, Ddn2]
        midpart = midpart / norm(midpart, Inf)
    end
    return nothing
end


function Cal_Obs_diag(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, pre_compute_env::pre_compute, Gates::Vector{String}, para::Dict{Symbol,Any}, get_op::Function; site=[1, 1, 2, 2])

end


# 计算单点观测量
function Cal_Obs_onsite(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, Gates::Vector{String}, para::Dict{Symbol,Any}, get_op::Function; site=[1, 1])
    x = site[1]
    y = site[2]
    M = ipeps[x, y]::TensorMap
    Mbar = ipepsbar[x, y]::TensorMap
    gate1 = swap_gate(space(M)[1], space(Mbar)[2]; Eltype=eltype(M))
    gate2 = swap_gate(space(M)[5], space(Mbar)[4]; Eltype=eltype(M))

    @tensor opt = true ψ□ψ[pup, pdn] :=
        envs[x, y].corner.lt[toT, toL] * envs[x, y].transfer.t[toT, toRT, toMtup, toMtdn] *
        envs[x, y].transfer.l[toL, toLB, toMlup, toMldn] * gate1[toMlup, toMtdn, toMlupin, toMtdnin] *
        M[toMlupin, toMtup, pup, toMrup, toMbupin] * Mbar[toMldn, toMtdnin, pdn, toMrdnin, toMbdn] *
        gate2[toMbup, toMrdn, toMbupin, toMrdnin] * envs[x, y].corner.rt[toRT, toR] *
        envs[x, y].corner.lb[toLB, toB] * envs[x, y].transfer.r[toMrup, toMrdn, toR, toRB] *
        envs[x, y].transfer.b[toB, toMbup, toMbdn, btoRB] * envs[x, y].corner.rb[btoRB, toRB]

    @tensor nrm = ψ□ψ[p, p]

    Obs_onsite = Dict{String,Number}()
    for gate in Gates
        OpL, OpR = get_op(gate, para; Nop=2)
        @tensor OnsiteOp[pdn, pup] := OpL[pdn, pin, a] * OpR[a, pin, pup]
        @tensor val = ψ□ψ[pup, pdn] * OnsiteOp[pdn, pup]
        Obs_onsite[gate] = val / nrm
    end

    return Obs_onsite
end