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

    return ncon((CapUp, M, Mbar, gate1, gate2), (idxA, [2, 6, 1, -rk - 1, 4], [-rk, 3, 1, 5, -rk - 4], [-rk + 1, 7, 2, 3], [-rk - 3, -rk - 2, 4, 5]))
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

    return ncon((CapL, M, Mbar, gate1, gate2), (idxA, [2, -rk + 1, 1, -rk - 3, 4], [7, 3, 1, 5, -rk - 2], [6, -rk, 2, 3], [-rk - 1, -rk - 4, 4, 5]))
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
    rslt = Dict{String,Matrix}()
    for gate in Gates
        OpL, OpR = get_op(gate, para; Nop=2)
        @tensor OnsiteOp[pup, pdn] := OpL[] * OpR[]

        Mat = Matrix{Number}(undef, 0, 5)

        for (ind1, val1) in enumerate(CartesianIndices((Lx, Ly)))
            (x1, y1) = Tuple(val1)
            for (ind2, val2) in enumerate(CartesianIndices((Lx, Ly)))
                (x2, y2) = Tuple(val2)
                if ind1 == ind2
                    Obs_onsite = Cal_Obs_1site(ipeps, ipepsbar, envs, [gate], para; site=[x1, x2], get_op)   # TODO: 这里再改改，先收缩算符，高效一点
                    Mat = vcat(Mat, [x1, y1, x2, y2, get(Obs_onsite, gate, NaN)])
                elseif x1 == x2
                    # 横向关联

                elseif y1 == y2
                    # 纵向关联

                else
                    # 斜对角关联
                end
            end
        end
        rslt[gate] = Mat
    end
    return rslt
end