include("../iPEPS_Fermionic/swap_gate.jl")
include("./get_proj.jl")
include("./apply_proj.jl")

"""
    CTMRG!(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, χ::Int, Nit::Int; parallel=false, threshold=1e-8)

Perform CTMRG: Max number of iterations is `Nit` and environment dimension `χ`. 

Keyword arguments: 

`parallel = true` will use parallel CTMRG which is faster but consumes more memmory.

`threshold` checks the convergence of CTMRG iterations. 
If the the norm difference of the corner tensor spectrum between two successive CTM steps
convergs below the given value, CTMRG process will stop.

The Fermionic version also requires input of `ipepsbar` for efficiency.
"""
function CTMRG!(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, χ::Int, Nit::Int; parallel=false, threshold=1e-8)
    if parallel
        _CTMRG_parallel!(ipeps, ipepsbar, envs, χ, Nit; threshold=threshold)
    else
        _CTMRG_seq!(ipeps, ipepsbar, envs, χ, Nit; threshold=threshold)
    end
    return nothing
end

function _CTMRG_seq!(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, χ::Int, Nit::Int; threshold=1e-8)
    Lx = ipeps.Lx
    Ly = ipeps.Ly
    it = 1
    cornerSpec = Matrix{Array}(undef, Lx, Ly)  # 记录 corner tensor 的 SVD 谱. 顺序：Slt, Slb, Srt, Srb
    while it <= Nit
        println("============ CTMRG iteration $it / $Nit =======================")
        @time "Left env" for xx in 1:Lx
            error_List = update_env_left_2by2!(ipeps, ipepsbar, envs, xx, χ)
            println("Iteration $it, update left edge (contract column-$xx) truncation error $(maximum(error_List))")
        end
        @time "Top env" for yy in 1:Ly
            error_List = update_env_top_2by2!(ipeps, ipepsbar, envs, yy, χ)
            println("Iteration $it, update top edge (contract row-$yy) truncation error $(maximum(error_List))")
        end
        @time "Right env" for xx in Lx:-1:1
            error_List = update_env_right_2by2!(ipeps, ipepsbar, envs, xx, χ)
            println("Iteration $it, update right edge (contract column-$xx) truncation error $(maximum(error_List))")
        end
        @time "Bottom env" for yy in Ly:-1:1
            error_List = update_env_bottom_2by2!(ipeps, ipepsbar, envs, yy, χ)
            println("Iteration $it, update bottom edge (contract row-$yy) truncation error $(maximum(error_List))")
        end
        # convergence check
        @time "Convergence Check" nrmDiff = CTM_convCheck!(cornerSpec, envs, it)
        if nrmDiff < threshold
            println("CTMRG converged at iteration $it with norm difference $nrmDiff < threshold $threshold")
            println()
            flush(stdout)
            GC.gc()
            break
        else
            println("Continue CTMRG with norm difference $nrmDiff > threshold $threshold")
            println()
            flush(stdout)
            GC.gc()
            it += 1
        end
    end
    return nothing
end

function _CTMRG_parallel!(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, χ::Int, Nit::Int; threshold=1e-8)
    Lx = ipeps.Lx
    Ly = ipeps.Ly
    it = 1
    cornerSpec = Matrix{Array}(undef, Lx, Ly)  # 记录 corner tensor 的 SVD 谱. 顺序：Slt, Slb, Srt, Srb
    while it <= Nit
        println("============ CTMRG iteration $it / $Nit =======================")
        @time "Left-right env" for xx in 1:Lx
            @floop for lr in 1:2
                if lr == 1
                    error_List_l = update_env_left_2by2!(ipeps, ipepsbar, envs, xx, χ)
                    println("Update left edge (contract column-$xx) truncation error $(maximum(error_List_l))")
                else
                    error_List_r = update_env_right_2by2!(ipeps, ipepsbar, envs, Lx - xx + 1, χ)
                    println("Update right edge (contract column-$(Lx-xx+1)) truncation error $(maximum(error_List_r))")
                end
            end
            flush(stdout)
        end
        @time "Top-Bottom env" for yy in 1:Ly
            @floop for tb in 1:2
                if tb == 1
                    error_List_t = update_env_top_2by2!(ipeps, ipepsbar, envs, yy, χ)
                    println("Update top edge (contract row-$yy) truncation error $(maximum(error_List_t))")
                else
                    error_List_b = update_env_bottom_2by2!(ipeps, ipepsbar, envs, Ly - yy + 1, χ)
                    println("Update bottom edge (contract row-$(Ly-yy+1)) truncation error $(maximum(error_List_b))")
                end
            end
            flush(stdout)
        end
        # convergence check
        @time "Convergence Check" nrmDiff = CTM_convCheck!(cornerSpec, envs, it)
        if nrmDiff < threshold
            println("CTMRG converged at iteration $it with norm difference $nrmDiff < threshold $threshold")
            println()
            flush(stdout)
            GC.gc()
            break
        else
            println("Continue CTMRG with norm difference $nrmDiff > threshold $threshold")
            println()
            flush(stdout)
            GC.gc()
            it += 1
        end
    end
    return nothing
end


function CTM_convCheck!(cornerSpec::Matrix, envs::iPEPSenv, it::Int)
    SpecNew = similar(cornerSpec)
    Lx = envs.Lx
    Ly = envs.Ly
    @floop for val in CartesianIndices((Lx, Ly))
        (xx, yy) = Tuple(val)
        _, Slt, _ = tsvd(envs.corner[xx, yy].lt, ((1,), (2,)); trunc=notrunc(), alg=TensorKit.SVD())
        _, Slb, _ = tsvd(envs.corner[xx, yy].lb, ((1,), (2,)); trunc=notrunc(), alg=TensorKit.SVD())
        _, Srt, _ = tsvd(envs.corner[xx, yy].rt, ((1,), (2,)); trunc=notrunc(), alg=TensorKit.SVD())
        _, Srb, _ = tsvd(envs.corner[xx, yy].rb, ((1,), (2,)); trunc=notrunc(), alg=TensorKit.SVD())
        SpecNew[xx, yy] = map(x -> diag(convert(Array, x)), [Slt, Slb, Srt, Srb])
    end

    if it == 1  # 第一轮CTMRG初始化 cornerSpec 即可，不用比较
        cornerSpec = SpecNew
        return Inf
    else
        difference = abs.(SpecNew - cornerSpec)
        record = Matrix{Float64}(undef, Lx, Ly)
        @floop for val in CartesianIndices((Lx, Ly))
            (xx, yy) = Tuple(val)
            lt = norm(difference[xx, yy][1])
            lb = norm(difference[xx, yy][2])
            rt = norm(difference[xx, yy][3])
            rb = norm(difference[xx, yy][4])
            record[xx, yy] = maximum([lt, lb, rt, rb])
        end
        cornerSpec = SpecNew
        return maximum(record)
    end
end


# 整个过程都要关注交换门，以及收缩顺序！！！！！！！！！！！
"""
    收缩第`x`列，更新第`x+1`列的左侧环境\n
    以 2*2 元胞为例：
    C ----
    | ← proj2
    T
    | ← proj1
    T
    | ← proj2
    C -----
"""
function update_env_left_2by2!(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, x::Int, χ::Int)
    Ly = ipeps.Ly
    # 存储更新左侧环境需要的投影算符，矩阵第一列是上半部分proj ∇, 第二列是下半部分proj Δ
    proj_List = Matrix{TensorMap}(undef, Ly, 2)
    # 误差列表
    error_List = Vector{Float64}(undef, Ly)
    # ----------------- 先求proj ---------------------
    Threads.@threads for yy in 1:Ly
        projup, projdn, ϵ = get_proj_update_left(ipeps, ipepsbar, envs, x, yy, χ)
        proj_List[yy, 1] = projup
        proj_List[yy, 2] = projdn
        error_List[yy] = ϵ
    end
    # ------------------ 再更新环境 ----------------------
    Threads.@threads for yy in 1:Ly
        if yy == 1
            apply_proj_left!(ipeps, ipepsbar, envs, proj_List[Ly, 2], proj_List[yy, 1], x, yy)
            apply_proj_ltCorner_updateL!(envs, proj_List[Ly, 1], x, 1)
            apply_proj_lbCorner_updateL!(envs, proj_List[1, 2], x, 1)
        else
            apply_proj_left!(ipeps, ipepsbar, envs, proj_List[yy-1, 2], proj_List[yy, 1], x, yy)
            apply_proj_ltCorner_updateL!(envs, proj_List[yy-1, 1], x, yy)
            apply_proj_lbCorner_updateL!(envs, proj_List[yy, 2], x, yy)
        end
    end
    return error_List
end


function update_env_right_2by2!(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, x::Int, χ::Int)
    Ly = ipeps.Ly
    # 存储更新右侧环境需要的投影算符，矩阵第一列是上半部分proj ∇, 第二列是下半部分proj Δ
    proj_List = Matrix{TensorMap}(undef, Ly, 2)
    # 误差列表
    error_List = Vector{Float64}(undef, Ly)
    # ----------------- 先求proj ---------------------
    Threads.@threads for yy in 1:Ly
        # 注意这里，求右侧/下侧投影算符时后，基准点要偏离一列/一行。也就是下面的`x-1`
        projup, projdn, ϵ = get_proj_update_right(ipeps, ipepsbar, envs, x - 1, yy, χ)
        proj_List[yy, 1] = projup
        proj_List[yy, 2] = projdn
        error_List[yy] = ϵ
    end
    # ------------------ 再更新环境 ----------------------
    Threads.@threads for yy in 1:Ly
        if yy == 1
            apply_proj_right!(ipeps, ipepsbar, envs, proj_List[Ly, 2], proj_List[yy, 1], x, yy)
            apply_proj_rtCorner_updateR!(envs, proj_List[Ly, 1], x, yy)
            apply_proj_rbCorner_updateR!(envs, proj_List[1, 2], x, yy)
        else
            apply_proj_right!(ipeps, ipepsbar, envs, proj_List[yy-1, 2], proj_List[yy, 1], x, yy)
            apply_proj_rtCorner_updateR!(envs, proj_List[yy-1, 1], x, yy)
            apply_proj_rbCorner_updateR!(envs, proj_List[yy, 2], x, yy)
        end
    end
    return error_List
end


function update_env_top_2by2!(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, y::Int, χ::Int)
    Lx = ipeps.Lx
    # 存储更新上侧环境需要的投影算符，矩阵第一列是左半部分proj ▷, 第二列是右半部分proj ◁
    proj_List = Matrix{TensorMap}(undef, Lx, 2)
    # 误差列表
    error_List = Vector{Float64}(undef, Lx)
    # ----------------- 先求proj ---------------------
    Threads.@threads for xx in 1:Lx
        projleft, projright, ϵ = get_proj_update_top(ipeps, ipepsbar, envs, xx, y, χ)
        proj_List[xx, 1] = projleft
        proj_List[xx, 2] = projright
        error_List[xx] = ϵ
    end
    # ------------------ 再更新环境 ----------------------
    Threads.@threads for xx in 1:Lx
        if xx == 1
            apply_proj_top!(ipeps, ipepsbar, envs, proj_List[Lx, 2], proj_List[xx, 1], xx, y)
            apply_proj_ltCorner_updateT!(envs, proj_List[Lx, 1], xx, y)
            apply_proj_rtCorner_updateT!(envs, proj_List[1, 2], xx, y)
        else
            apply_proj_top!(ipeps, ipepsbar, envs, proj_List[xx-1, 2], proj_List[xx, 1], xx, y)
            apply_proj_ltCorner_updateT!(envs, proj_List[xx-1, 1], xx, y)
            apply_proj_rtCorner_updateT!(envs, proj_List[xx, 2], xx, y)
        end
    end
    return error_List
end


function update_env_bottom_2by2!(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, y::Int, χ::Int)
    Lx = ipeps.Lx
    # 存储更新上侧环境需要的投影算符，矩阵第一列是左半部分proj ▷, 第二列是右半部分proj ◁
    proj_List = Matrix{TensorMap}(undef, Lx, 2)
    # 误差列表
    error_List = Vector{Float64}(undef, Lx)
    # ----------------- 先求proj ---------------------
    Threads.@threads for xx in 1:Lx
        # 注意这里，求右侧/下侧投影算符时后，基准点要偏离一列/一行。也就是下面的`y-1`
        projleft, projright, ϵ = get_proj_update_bottom(ipeps, ipepsbar, envs, xx, y - 1, χ)
        proj_List[xx, 1] = projleft
        proj_List[xx, 2] = projright
        error_List[xx] = ϵ
    end
    # ------------------ 再更新环境 ----------------------
    Threads.@threads for xx in 1:Lx
        if xx == 1
            apply_proj_bottom!(ipeps, ipepsbar, envs, proj_List[Lx, 2], proj_List[xx, 1], xx, y)
            apply_proj_lbCorner_updateB!(envs, proj_List[Lx, 1], xx, y)
            apply_proj_rbCorner_updateB!(envs, proj_List[1, 2], xx, y)
        else
            apply_proj_bottom!(ipeps, ipepsbar, envs, proj_List[xx-1, 2], proj_List[xx, 1], xx, y)
            apply_proj_lbCorner_updateB!(envs, proj_List[xx-1, 1], xx, y)
            apply_proj_rbCorner_updateB!(envs, proj_List[xx, 2], xx, y)
        end
    end
    return error_List
end