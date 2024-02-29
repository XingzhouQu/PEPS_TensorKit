include("./get_proj.jl")
include("./apply_proj.jl")

function CTMRG!(ipeps::iPEPS, envs::iPEPSenv, χ::Int, Nit::Int)
    Lx = ipeps.Lx
    Ly = ipeps.Ly
    it = 1
    while it <= Nit
        println("============ CTMRG iteration $it / $Nit =======================")
        # 这里的顺序可能也对优化结果有影响，可以测试
        @time for xx in 1:Lx
            error_List = update_env_left_2by2!(ipeps, envs, xx, χ)
            println("Iteration $it, update left edge (contract column-$xx) truncation error $(maximum(error_List))")
        end
        GC.gc()
        @time for yy in 1:Ly
            error_List = update_env_top_2by2!(ipeps, envs, yy, χ)
            println("Iteration $it, update top edge (contract row-$yy) truncation error $(maximum(error_List))")
        end
        GC.gc()
        @time for xx in 1:Lx
            error_List = update_env_right_2by2!(ipeps, envs, xx, χ)
            println("Iteration $it, update right edge (contract column-$xx) truncation error $(maximum(error_List))")
        end
        GC.gc()
        @time for yy in 1:Ly
            error_List = update_env_bottom_2by2!(ipeps, envs, yy, χ)
            println("Iteration $it, update bottom edge (contract row-$yy) truncation error $(maximum(error_List))")
        end
        GC.gc()
        println()
        flush(stdout)
        it += 1
    end
    return nothing
end


# 整个过程都要关注交换门，以及收缩顺序！！！！！！！！！！！
"""
    更新第 `x` 列元胞的左侧环境\n
    以 2*2 元胞为例：
    C ----
    | ← proj2
    T
    | ← proj1
    T
    | ← proj2
    C -----
"""
function update_env_left_2by2!(ipeps::iPEPS, envs::iPEPSenv, x::Int, χ::Int)
    Ly = ipeps.Ly
    # 存储更新左侧环境需要的投影算符，矩阵第一列是上半部分proj ∇, 第二列是下半部分proj Δ
    proj_List = Matrix{TensorMap}(undef, Ly, 2)
    # 误差列表
    error_List = Vector{Float64}(undef, Ly)
    # ----------------- 先求proj ---------------------
    for yy in 1:Ly
        projup, projdn, ϵ = get_proj_update_LR(ipeps, envs, x, yy, χ; dir="left")
        proj_List[yy, 1] = projup
        proj_List[yy, 2] = projdn
        error_List[yy] = ϵ
    end
    # ------------------ 再更新环境 ----------------------
    for yy in 1:Ly
        if yy == 1
            apply_proj_left!(ipeps, envs, proj_List[Ly, 2], proj_List[yy, 1], x, yy)
            apply_proj_ltCorner_updateL!(envs, proj_List[Ly, 1], x, 1)
            apply_proj_lbCorner_updateL!(envs, proj_List[1, 2], x, 1)
        else
            apply_proj_left!(ipeps, envs, proj_List[yy-1, 2], proj_List[yy, 1], x, yy)
            apply_proj_ltCorner_updateL!(envs, proj_List[yy-1, 1], x, yy)
            apply_proj_lbCorner_updateL!(envs, proj_List[yy, 2], x, yy)
        end
    end
    # 返回误差
    return error_List
end


function update_env_right_2by2!(ipeps::iPEPS, envs::iPEPSenv, x::Int, χ::Int)
    Ly = ipeps.Ly
    # 存储更新右侧环境需要的投影算符，矩阵第一列是上半部分proj ∇, 第二列是下半部分proj Δ
    proj_List = Matrix{TensorMap}(undef, Ly, 2)
    # 误差列表
    error_List = Vector{Float64}(undef, Ly)
    # ----------------- 先求proj ---------------------
    for yy in 1:Ly
        # 注意这里，求右侧/下侧投影算符时后，基准点要偏离一列/一行。也就是下面的`x-1`
        projup, projdn, ϵ = get_proj_update_LR(ipeps, envs, x-1, yy, χ; dir="right")
        proj_List[yy, 1] = projup
        proj_List[yy, 2] = projdn
        error_List[yy] = ϵ
    end
    # ------------------ 再更新环境 ----------------------
    for yy in 1:Ly
        if yy == 1
            apply_proj_right!(ipeps, envs, proj_List[Ly, 2], proj_List[yy, 1], x, yy)
            apply_proj_rtCorner_updateR!(envs, proj_List[Ly, 1], x, 1)
            apply_proj_rbCorner_updateR!(envs, proj_List[1, 2], x, 1)
        else
            apply_proj_right!(ipeps, envs, proj_List[yy-1, 2], proj_List[yy, 1], x, yy)
            apply_proj_rtCorner_updateR!(envs, proj_List[yy-1, 1], x, yy)
            apply_proj_rbCorner_updateR!(envs, proj_List[yy, 2], x, yy)
        end
    end
    # 返回误差
    return error_List
end


function update_env_top_2by2!(ipeps::iPEPS, envs::iPEPSenv, y::Int, χ::Int)
    Lx = ipeps.Lx
    # 存储更新上侧环境需要的投影算符，矩阵第一列是左半部分proj ▷, 第二列是右半部分proj ◁
    proj_List = Matrix{TensorMap}(undef, Lx, 2)
    # 误差列表
    error_List = Vector{Float64}(undef, Lx)
    # ----------------- 先求proj ---------------------
    for xx in 1:Lx
        projleft, projright, ϵ = get_proj_update_UD(ipeps, envs, xx, y, χ; dir="up")
        proj_List[xx, 1] = projleft
        proj_List[xx, 2] = projright
        error_List[xx] = ϵ
    end
    # ------------------ 再更新环境 ----------------------
    for xx in 1:Lx
        if xx == 1
            apply_proj_top!(ipeps, envs, proj_List[Lx, 2], proj_List[xx, 1], xx, y)
            apply_proj_ltCorner_updateT!(envs, proj_List[Lx, 1], 1, y)
            apply_proj_rtCorner_updateT!(envs, proj_List[1, 2], 1, y)
        else
            apply_proj_top!(ipeps, envs, proj_List[xx-1, 2], proj_List[xx, 1], xx, y)
            apply_proj_ltCorner_updateT!(envs, proj_List[xx-1, 1], xx, y)
            apply_proj_rtCorner_updateT!(envs, proj_List[xx, 2], xx, y)
        end
    end
    # 返回误差
    return error_List
end


function update_env_bottom_2by2!(ipeps::iPEPS, envs::iPEPSenv, y::Int, χ::Int)
    Lx = ipeps.Lx
    # 存储更新上侧环境需要的投影算符，矩阵第一列是左半部分proj ▷, 第二列是右半部分proj ◁
    proj_List = Matrix{TensorMap}(undef, Lx, 2)
    # 误差列表
    error_List = Vector{Float64}(undef, Lx)
    # ----------------- 先求proj ---------------------
    for xx in 1:Lx
        # 注意这里，求右侧/下侧投影算符时后，基准点要偏离一列/一行。也就是下面的`y-1`
        projleft, projright, ϵ = get_proj_update_UD(ipeps, envs, xx, y-1, χ; dir="dn")
        proj_List[xx, 1] = projleft
        proj_List[xx, 2] = projright
        error_List[xx] = ϵ
    end
    # ------------------ 再更新环境 ----------------------
    for xx in 1:Lx
        if xx == 1
            apply_proj_bottom!(ipeps, envs, proj_List[Lx, 2], proj_List[xx, 1], xx, y)
            apply_proj_lbCorner_updateB!(envs, proj_List[Lx, 1], 1, y)
            apply_proj_rbCorner_updateB!(envs, proj_List[1, 2], 1, y)
        else
            apply_proj_bottom!(ipeps, envs, proj_List[xx-1, 2], proj_List[xx, 1], xx, y)
            apply_proj_lbCorner_updateB!(envs, proj_List[xx-1, 1], xx, y)
            apply_proj_rbCorner_updateB!(envs, proj_List[xx, 2], xx, y)
        end
    end
    # 返回误差
    return error_List
end