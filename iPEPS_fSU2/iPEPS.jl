import Base.getindex
include("./ini_env.jl")

struct iPEPS
    Ms::AbstractMatrix{AbstractTensorMap}
    Lx::Int
    Ly::Int
    # 按列初始化
    function iPEPS(Ms::AbstractVector{T}, Lx::Int, Ly::Int) where {T}
        println("Initializing iPEPS with eltype $(mapreduce(eltype, promote_type, storagetype.(Ms)))")
        Ms = reshape(Ms, (Lx, Ly))
        return new(Ms, Lx, Ly)
    end
    function iPEPS(Ms::Matrix{TensorMap}, Lx::Int, Ly::Int)
        return new(Ms, Lx, Ly)
    end
end

# 内部类，从ipeps构造角矩阵和边转移矩阵
mutable struct _corner
    lt::AbstractTensorMap
    lb::AbstractTensorMap
    rt::AbstractTensorMap
    rb::AbstractTensorMap
    function _corner(ipeps::iPEPS, x::Int, y::Int)
        return new(ini_lt_corner(ipeps[x-1, y-1]), ini_lb_corner(ipeps[x-1, y+1]),
            ini_rt_corner(ipeps[x+1, y-1]), ini_rb_corner(ipeps[x+1, y+1]))
    end
end

mutable struct _transfer
    l::AbstractTensorMap
    r::AbstractTensorMap
    t::AbstractTensorMap
    b::AbstractTensorMap
    function _transfer(ipeps::iPEPS, x::Int, y::Int)
        return new(ini_l_transfer(ipeps[x-1, y]), ini_r_transfer(ipeps[x+1, y]),
            ini_t_transfer(ipeps[x, y-1]), ini_b_transfer(ipeps[x, y+1]))
    end
end

# ===============================================
# Wrapper
struct _iPEPSenv
    corner::_corner
    transfer::_transfer
    function _iPEPSenv(ipeps::iPEPS, x::Int, y::Int)
        return new(_corner(ipeps, x, y), _transfer(ipeps, x, y))
    end
end

"""
初始化 iPEPS 环境张量.\n
内部构造方法 `iPEPSenv(ipeps::iPEPS)`.\n
用例：
```
envs = iPEPSenv(ipeps)
envs[1,1].corner.rt
envs[2,1].transfer.l
```
"""
struct iPEPSenv
    Envs::AbstractMatrix{_iPEPSenv}
    Lx::Int
    Ly::Int
    function iPEPSenv(ipeps::iPEPS)
        Lx = ipeps.Lx
        Ly = ipeps.Ly
        envs = Matrix{_iPEPSenv}(undef, Lx, Ly)
        for x in 1:Lx, y in 1:Ly
            envs[x, y] = _iPEPSenv(ipeps, x, y)
        end
        return new(envs, Lx, Ly)
    end
end

# ==================== ΓΛ notaion======================
"""
每个点的Γ和上下左右的四个bond matrix.
注意会有重复！！！ 在更新的时候每次都要改动两个！！！
"""
mutable struct _iPEPSΓΛ
    Γ::TensorMap
    l::TensorMap
    r::TensorMap
    t::TensorMap
    b::TensorMap

    function _iPEPSΓΛ(ipepst::TensorMap, s::TensorMap)  # 随便初始化一下，用于iPEPSΓΛ的构造。内部类不对外开放
        l = r = t = b = s
        return new(ipepst, l, r, t, b)
    end
    function _iPEPSΓΛ(A::TensorMap, slr::TensorMap, stb::TensorMap)  # iPEPSΓΛ的随机初始化。内部类不对外开放
        return new(A, slr, slr, stb, stb)
    end
end


"""
构造方法：\n
`ipepsΓΛ =  iPEPSΓΛ(pspace::VectorSpace, aspacelr::VectorSpace, aspacetb::VectorSpace, Lx::Int, Ly::Int; dtype=ComplexF64)`

用法：\n
`ipepsΓΛ[x, y].Γ  →  Γ`
`ipepsΓΛ[x, y].l`
`ipepsΓΛ[x, y].r`
`ipepsΓΛ[x, y].t`
`ipepsΓΛ[x, y].b`

    ——————>  x
    |
    |
    v
     y 
"""
struct iPEPSΓΛ
    ΓΛ::Matrix{_iPEPSΓΛ}
    Lx::Int
    Ly::Int

    function iPEPSΓΛ(γλ::Matrix{_iPEPSΓΛ}, Lx, Ly)
        return new(γλ, Lx, Ly)
    end
    # 初始化 ΓΛ 形式的 iPEPS. 注意左上不是对偶空间，右下的指标在对偶空间
    # tmp[l, t, p; r, b] := ipepsΓΛ[1,1].Γ[l, t, p, r, bin] * ipepsΓΛ[1,1].b[bin, b]
    # tmp[l, t, p; r, b] := ipepsΓΛ[1,1].Γ[lin, t, p, r, b] * ipepsΓΛ[1,1].l[l, lin]
    function iPEPSΓΛ(pspace::VectorSpace, aspacelr::VectorSpace, aspacetb::VectorSpace, Lx::Int, Ly::Int; dtype=ComplexF64)
        γλ = Matrix{_iPEPSΓΛ}(undef, Lx, Ly)
        for xx in 1:Lx, yy in 1:Ly
            # 这里以后可以改用randisometry初始化。现在版本randisometry有bug?
            tmp = TensorMap(randn, dtype, aspacelr ⊗ aspacetb ⊗ pspace, aspacelr ⊗ aspacetb)
            γλ[xx, yy] = _iPEPSΓΛ(tmp / norm(tmp), id(aspacelr), id(aspacetb))
        end
        # fill!(γλ, ini)  # 注意：这里不能用 fill! 初始化，这样会把所有的矩阵元都对应同一个引用，改一个就是在改所有！！！
        return new(γλ, Lx, Ly)
    end
end


function updateγλ_ud!(γλ::Matrix{_iPEPSΓΛ}, xx::Int, yy::Int)
    Ly = size(γλ, 2)
    if yy == Ly
        A = γλ[xx, Ly].Γ
        B = γλ[xx, 1].Γ
    else
        A = γλ[xx, yy].Γ
        B = γλ[xx, yy+1].Γ
    end
    @tensor AB[la, ta, pa, ra, lb, pb, rb, bb] := A[la, ta, pa, ra, in] * B[lb, in, pb, rb, bb]
    U, S, Vdag, _ = tsvd(AB, ((1, 2, 3, 4), (5, 6, 7, 8)), trunc=notrunc())
    γλ[xx, yy].Γ = U
    γλ[xx, yy].b = S
    if yy == Ly
        γλ[xx, 1].Γ = permute(Vdag, (2, 1, 3), (4, 5))
        γλ[xx, 1].t = S
    else
        γλ[xx, yy+1].Γ = permute(Vdag, (2, 1, 3), (4, 5))
        γλ[xx, yy+1].t = S
    end
    return nothing
end

function updateγλ_lr!(γλ::Matrix{_iPEPSΓΛ}, xx::Int, yy::Int)
    Lx = size(γλ, 1)
    if xx == Lx
        A = γλ[Lx, yy].Γ
        B = γλ[1, yy].Γ
    else
        A = γλ[xx, yy].Γ
        B = γλ[xx+1, yy].Γ
    end
    @tensor AB[la, ta, pa, ba, tb, pb, rb, bb] := A[la, ta, pa, in, ba] * B[in, tb, pb, rb, bb]
    U, S, Vdag, _ = tsvd(AB, ((1, 2, 3, 4), (5, 6, 7, 8)), trunc=notrunc())
    γλ[xx, yy].Γ = permute(U, (1, 2, 3), (5, 4))
    γλ[xx, yy].r = S
    if xx == Lx
        γλ[xx, 1].Γ = Vdag
        γλ[xx, 1].l = S
    else
        γλ[xx+1, yy].Γ = Vdag
        γλ[xx+1, yy].l = S
    end
    return nothing
end

# ======================= 两种形式互相转化，外部构造方法 ====================
# 从 ΓΛ 形式转换为正常形式
function iPEPS(ipeps::iPEPSΓΛ)
    Lx = ipeps.Lx
    Ly = ipeps.Ly
    Ms = Matrix{TensorMap}(undef, Lx, Ly)
    for x in 1:Lx, y in 1:Ly
        # println("x=$x, y=$y, space is $(space(ipeps[x, y].Γ))")
        @tensor tmp[l, t, p; r, b] := ipeps[x, y].Γ[le, te, p, re, be] * sqrt(ipeps[x, y].l)[l, le] *
                                      sqrt(ipeps[x, y].t)[t, te] * sqrt(ipeps[x, y].r)[re, r] * sqrt(ipeps[x, y].b)[be, b]
        # tmp = ipeps[x, y].Γ * sqrt(ipeps[x, y].l) * sqrt(ipeps[x, y].t) * sqrt(ipeps[x, y].r) * sqrt(ipeps[x, y].b)
        Ms[x, y] = tmp
    end
    return iPEPS(Ms, Lx, Ly)
end

# 转换正常的iPEPS为 ΓΛ 形式，用处不大，也不知道对不对
function iPEPSΓΛ(ipeps::iPEPS)
    Lx = ipeps.Lx
    Ly = ipeps.Ly
    # 初始化, 先填充避免undef无法访问
    γλ = Matrix{_iPEPSΓΛ}(undef, Lx, Ly)
    for xx in 1:Lx, yy in 1:Ly
        γλ[xx, yy] = _iPEPSΓΛ(ipeps[xx, yy], id(space(ipeps[1, 1])[3]))
    end
    # 对每一个xx遍历yy, 纵向更新Γ 和 Λ.
    for xx in 1:Lx, yy in 1:Ly
        updateγλ_ud!(γλ, xx, yy)
    end
    # 对每一个yy遍历xx, 横向更新Γ 和 Λ.
    for yy in 1:Ly, xx in 1:Lx
        updateγλ_lr!(γλ, xx, yy)
    end
    return new(γλ, Lx, Ly)
end

# ======= Reload getindex ========================
"""
可以直接用iPEPS[x, y]索引对应的tensor.\n
若 [x, y] 超出元胞周期，则自动回归到周期内.
"""
function getindex(ipeps::iPEPS, idx::Int, idy::Int)
    Lx = ipeps.Lx
    Ly = ipeps.Ly
    idx = idx - Int(ceil(idx / Lx) - 1) * Lx
    idy = idy - Int(ceil(idy / Ly) - 1) * Ly
    return getindex(ipeps.Ms, idx, idy)
end

function getindex(envs::iPEPSenv, idx::Int, idy::Int)
    Lx = envs.Lx
    Ly = envs.Ly
    idx = idx - Int(ceil(idx / Lx) - 1) * Lx
    idy = idy - Int(ceil(idy / Ly) - 1) * Ly
    return getindex(envs.Envs, idx, idy)
end

"""
可以直接用ipepsΓΛ[x, y]索引对应的tensor.\n
若 [x, y] 超出元胞周期，则自动回归到周期内.
"""
function getindex(ipepsΓΛ::iPEPSΓΛ, idx::Int, idy::Int)
    Lx = ipepsΓΛ.Lx
    Ly = ipepsΓΛ.Ly
    idx = idx - Int(ceil(idx / Lx) - 1) * Lx
    idy = idy - Int(ceil(idy / Ly) - 1) * Ly
    return getindex(ipepsΓΛ.ΓΛ, idx, idy)
end

include("./util.jl")
