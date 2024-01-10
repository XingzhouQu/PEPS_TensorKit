import Base.getindex

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
内部构造方法：\n
`ipepsΓΛ =  iPEPSΓΛ(ipeps::iPEPS)`

用法：\n
`ipepsΓΛ[x, y].Γ  →  Γ`
`ipepsΓΛ[x, y].l`
`ipepsΓΛ[x, y].r`
`ipepsΓΛ[x, y].t`
`ipepsΓΛ[x, y].b`
"""
struct iPEPSΓΛ
    ΓΛ::Matrix{_iPEPSΓΛ}
    Lx::Int
    Ly::Int
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
    # 初始化 ΓΛ 形式的 iPEPS. 
    # tmp[l, t, p; r, b] := ipepsΓΛ[1,1].Γ[l, t, p, r, bin] * ipepsΓΛ[1,1].b[bin, b]
    # tmp[l, t, p; r, b] := ipepsΓΛ[1,1].Γ[lin, t, p, r, b] * ipepsΓΛ[1,1].l[l, lin]
    function iPEPSΓΛ(pspace::VectorSpace, aspacelr::VectorSpace, aspacetb::VectorSpace, Lx::Int, Ly::Int; dtype=ComplexF64)
        γλ = Matrix{_iPEPSΓΛ}(undef, Lx, Ly)
        A = TensorMap(randn, dtype, aspacelr ⊗ aspacetb ⊗ pspace, aspacelr ⊗ aspacetb)
        slr = id(aspacelr)
        stb = id(aspacetb)
        ini = _iPEPSΓΛ(A, slr, stb)
        fill!(γλ, ini)
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


# ======= Reload getindex ========================
"""
可以直接用ipepsΓΛ[x, y]索引对应的tensor.\n
若 [x, y] 超出元胞周期，则自动回归到周期内.
"""
function getindex(ipepsΓΛ::iPEPSΓΛ, idx::Int, idy::Int)
    Lx = ipepsΓΛ.Lx
    Ly = ipepsΓΛ.Ly
    idx = idx - Int(ceil(idx / Lx) - 1)
    idy = idy - Int(ceil(idy / Ly) - 1)
    return getindex(ipepsΓΛ.ΓΛ, idx, idy)
end