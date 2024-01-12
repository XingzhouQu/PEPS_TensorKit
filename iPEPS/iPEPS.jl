import Base.getindex
include("./swap_gate.jl")

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
    # 从 ΓΛ 形式转换为正常形式
    function iPEPS(ipeps::iPEPSΓΛ)
        Lx = ipeps.Lx
        Ly = ipeps.Ly
        Ms = Matrix{TensorMap}(undef, Lx, Ly)
        for xx in 1:Lx, yy in 1:Ly
            @tensor tmp[l, t, p; r, b] := ipeps[x, y].Γ[le, te, p, re, be] * sqrt(ipeps[x, y].l)[l, le] *
                                          sqrt(ipeps[x, y].t)[t, te] * sqrt(ipeps[x, y].r)[re, r] * sqrt(ipeps[x, y].b)[be, b]
            Ms[xx, yy] = tmp
        end
        return new(Ms, Lx, Ly)
    end
end

# 内部类，从ipeps构造角矩阵和边转移矩阵
mutable struct _corner
    lt::AbstractTensorMap
    lb::AbstractTensorMap
    rt::AbstractTensorMap
    rb::AbstractTensorMap
    function _corner(A::AbstractTensorMap)
        return new(ini_lt_corner(A), ini_lb_corner(A), ini_rt_corner(A), ini_rb_corner(A))
    end
end

mutable struct _transfer
    l::AbstractTensorMap
    r::AbstractTensorMap
    t::AbstractTensorMap
    b::AbstractTensorMap
    function _transfer(A::AbstractTensorMap)
        return new(ini_l_transfer(A), ini_r_transfer(A), ini_t_transfer(A), ini_b_transfer(A))
    end
end

# ===========================================
"""
left-top corner environment tensor.

Space: lt:() ← (rup, bup, rdn, bdn). 【up layer for M and dn layer for Mbar】
"""
function ini_lt_corner(A::AbstractTensorMap)
    Abar = bar(A)
    gate1 = swap_gate(space(A)[1], space(A)[2])
    gate2 = swap_gate(space(Abar)[4], space(A)[5])
    @tensor lt[(); (rup, bup, rdn, bdn)] := (gate1[ldnin, tdnin, lupin, tupin] * A[lupin, tupin, p, rup, bupin]) *
                                            (gate2[rdn, bup, rdnin, bupin] * Abar[ldnin, tdnin, p, rdnin, bdn])
    frspace = fuse(space(lt)[1], space(lt)[3])
    isor = isometry(dual(frspace), space(lt)[1] ⊗ space(lt)[3])
    fbspace = fuse(space(lt)[2], space(lt)[4])
    isob = isometry(dual(fbspace), space(lt)[2] ⊗ space(lt)[4])
    @tensor ltfuse[(rχ, bχ); ()] := (isor[rχ, rupχin, rdnχin] * lt[rupχin, bupχin, rdnχin, bdnχin]) * isob[bχ, bupχin, bdnχin]
    return ltfuse
end


"""
left-bottom corner environment tensor.

Space: lb:(tdn, tup) ← (rup, rdn). 【up layer for M and dn layer for Mbar】
"""
function ini_lb_corner(A::AbstractTensorMap)
    Abar = bar(A)
    gate1 = swap_gate(space(Abar)[1], space(Abar)[2])
    gate2 = swap_gate(space(Abar)[4], space(Abar)[5])
    @tensor lb[tdn, tup; rup, rdn] :=
        ((gate1[lupin, tdn, ldnin, tdnin] * Abar[ldnin, tdnin, p, rdnin, bdnin]) * gate2[rdn, bupin, rdnin, bdnin]) *
        A[lupin, tup, p, rup, bupin]
    ftspace = fuse(space(lb)[2], space(lb)[1])
    isot = isometry(ftspace, space(lb)[2] ⊗ space(lb)[1])
    frspace = fuse(space(lb)[3], space(lb)[4])
    isor = isometry(dual(frspace), space(lb)[3] ⊗ space(lb)[4])
    @tensor lbfuse[(tχ, rχ); ()] := (isot[tχ, tupχin, tdnχin] * lb[tdnχin, tupχin, rupχin, rdnχin]) * isor[rχ, rupχin, rdnχin]
    return lbfuse
end


"""
right-top corner environment tensor.

Space: rt:(lup, ldn) ← (bup bdn). 【up layer for M and dn layer for Mbar】
"""
function ini_rt_corner(A::AbstractTensorMap)
    Abar = bar(A)
    gate1 = swap_gate(space(A)[1], space(A)[2])
    gate2 = swap_gate(space(A)[4], space(A)[5])
    @tensor rt[lup ldn; bup bdn] := (gate1[lup, tdnin, lupin, tupin] * A[lupin, tupin, p, rupin, bupin]) *
                                    (gate2[rdnin, bup, rupin, bupin] * Abar[ldn, tdnin, p, rdnin, bdn])
    flspace = fuse(space(rt)[1], space(rt)[2])
    isol = isometry(flspace, space(rt)[1] ⊗ space(rt)[2])
    fbspace = fuse(space(rt)[3], space(rt)[4])
    isob = isometry(dual(fbspace), space(rt)[3] ⊗ space(rt)[4])
    @tensor rtfuse[(lχ, bχ); ()] := (isol[lχ, lupχin, ldnχin] * rt[lupχin, ldnχin, bupχin, bdnχin]) * isob[bχ, bupχin, bdnχin]
    return rtfuse
end


"""
right-bottom corner environment tensor.

Space: rb:(lup, ldn, tup, tdn) ← (). 【up layer for M and dn layer for Mbar】
"""
function ini_rb_corner(A::AbstractTensorMap)
    Abar = bar(A)
    gate1 = swap_gate(space(A)[1], space(A)[2])
    gate2 = swap_gate(space(A)[4], space(A)[5])
    @tensor rb[(lup, ldn, tup, tdn); ()] := (gate1[lup, tdnin, lupin, tdn] * A[lupin, tup, p, rupin, bupin]) *
                                            (gate2[rdnin, bdnin, rupin, bupin] * Abar[ldn, tdnin, p, rdnin, bdnin])
    flspace = fuse(space(rb)[1], space(rb)[2])
    isol = isometry(flspace, space(rb)[1] ⊗ space(rb)[2])
    ftspace = fuse(space(rb)[3], space(rb)[4])
    isot = isometry(ftspace, space(rb)[3] ⊗ space(rb)[4])
    @tensor rbfuse[(); (lχ, tχ)] := (isol[lχ, lupχin, ldnχin] * rb[lupχin, ldnχin, tupχin, tdnχin]) * isot[tχ, tupχin, tdnχin]
    return rbfuse
end

# =========================================
"""
left edge transfer tensor.

Return: `l::Vector{TensorMap}`. `l[i]`标记第`i`行的，从上到下
"""
function ini_l_transfer(A::AbstractTensorMap)
    Abar = bar(A)
    gate1 = swap_gate(space(A)[1], space(A)[2])
    gate2 = swap_gate(space(Abar)[4], space(A)[5])
    @tensor tmp[(tup, tdn); (rup, rdn, bup, bdn)] := (gate1[ldnin, tdnin, lupin, tdn] * A[lupin, tup, p, rup, bupin]) *
                                                     (gate2[rdn, bup, rdnin, bupin] * Abar[ldnin, tdnin, p, rdnin, bdn])
    ftspace = fuse(space(tmp)[1], space(tmp)[2])
    isot = isometry(ftspace, space(tmp)[1] ⊗ space(tmp)[2])
    fbspace = fuse(space(tmp)[5], space(tmp)[6])
    isob = isometry(dual(fbspace), space(tmp)[5] ⊗ space(tmp)[6])
    @tensor tmpfuse[(tχ, bχ); (rup, rdn)] := (isot[tχ, tupχin, tdnχin] * tmp[tupχin, tdnχin, rup, rdn, bupχin, bdnχin]) * isob[bχ, bupχin, bdnχin]
    return tmpfuse
end

"""
right edge transfer tensor.

Return: `r::Vector{TensorMap}`. `r[i]`标记第`i`行的，从上到下
"""
function ini_r_transfer(A::AbstractTensorMap)
    Abar = bar(A)
    gate1 = swap_gate(space(A)[1], space(A)[2])
    gate2 = swap_gate(space(A)[4], space(A)[5])
    @tensor tmp[(lup, ldn, tup, tdn); (bup, bdn)] := (gate1[lup, tdnin, lupin, tdn] * A[lupin, tup, p, rupin, bupin]) *
                                                     (gate2[rdnin, bup, rupin, bupin] * Abar[ldn, tdnin, p, rdnin, bdn])
    ftspace = fuse(space(tmp)[3], space(tmp)[4])
    isot = isometry(ftspace, space(tmp)[3] ⊗ space(tmp)[4])
    fbspace = fuse(space(tmp)[5], space(tmp)[6])
    isob = isometry(dual(fbspace), space(tmp)[5] ⊗ space(tmp)[6])
    @tensor tmpfuse[(lup, ldn, tχ, bχ); ()] := (isot[tχ, tupχin, tdnχin] * tmp[lup, ldn, tupχin, tdnχin, bupχin, bdnχin]) * isob[bχ, bupχin, bdnχin]
    return tmpfuse
end

"""
top edge transfer tensor.

Return: `t::Vector{TensorMap}`. `t[i]`标记第`i`行的，从上到下
"""
function ini_t_transfer(A::AbstractTensorMap)
    Abar = bar(A)
    gate1 = swap_gate(space(A)[1], space(A)[2])
    gate2 = swap_gate(space(Abar)[4], space(A)[5])
    @tensor tmp[(lup, ldn); (rup, rdn, bup, bdn)] := (gate1[lup, tdnin, lupin, tupin] * A[lupin, tupin, p, rup, bupin]) *
                                                     (gate2[rdn, bup, rdnin, bupin] * Abar[ldn, tdnin, p, rdnin, bdn])
    flspace = fuse(space(tmp)[1], space(tmp)[2])
    isol = isometry(flspace, space(tmp)[1] ⊗ space(tmp)[2])
    frspace = fuse(space(tmp)[3], space(tmp)[4])
    isor = isometry(dual(frspace), space(tmp)[3] ⊗ space(tmp)[4])
    @tensor tmpfuse[(lχ, rχ); (bup, bdn)] := (isol[lχ, lupχin, ldnχin] * tmp[lupχin, ldnχin, rupχin, rdnχin, bup, bdn]) * isor[rχ, rupχin, rdnχin]
    return tmpfuse
end

"""
bottom edge transfer tensor.

Return: `b::Vector{TensorMap}`. `b[i]`标记第`i`行的，从上到下
"""
function ini_b_transfer(A::AbstractTensorMap)
    Abar = bar(A)
    gate1 = swap_gate(space(Abar)[1], space(Abar)[2])
    gate2 = swap_gate(space(Abar)[4], space(Abar)[5])
    @tensor tmp[(lup, ldn, tup, tdn); (rup, rdn)] :=
        ((gate1[lupin, tdn, lup, tdnin] * Abar[ldn, tdnin, p, rdnin, bdnin]) * gate2[rdn, bupin, rdnin, bdnin]) *
        A[lupin, tup, p, rup, bupin]

    flspace = fuse(space(tmp)[1], space(tmp)[2])
    isol = isometry(flspace, space(tmp)[1] ⊗ space(tmp)[2])
    frspace = fuse(space(tmp)[5], space(tmp)[6])
    isor = isometry(dual(frspace), space(tmp)[5] ⊗ space(tmp)[6])
    @tensor tmpfuse[(lχ, tup, tdn, rχ); ()] := (isol[lχ, lupχin, ldnχin] * tmp[lupχin, ldnχin, tup, tdn, rupχin, rdnχin]) * isor[rχ, rupχin, rdnχin]
    return tmpfuse
end

# ===============================================
# Wrapper
struct _iPEPSenv
    corner::_corner
    transfer::_transfer
    function _iPEPSenv(t::AbstractTensorMap)
        return new(_corner(t), _transfer(t))
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
            tensor = ipeps[x, y]
            envs[x, y] = _iPEPSenv(tensor)
        end
        return new(envs, Lx, Ly)
    end
end

# ======= Reload getindex ========================
"""
可以直接用iPEPS[x, y]索引对应的tensor.\n
若 [x, y] 超出元胞周期，则自动回归到周期内.
"""
function getindex(ipeps::iPEPS, idx::Int, idy::Int)
    Lx = ipeps.Lx
    Ly = ipeps.Ly
    idx = idx - Int(ceil(idx / Lx) - 1)
    idy = idy - Int(ceil(idy / Ly) - 1)
    return getindex(ipeps.Ms, idx, idy)
end

function getindex(envs::iPEPSenv, idx::Int, idy::Int)
    Lx = envs.Lx
    Ly = envs.Ly
    idx = idx - Int(ceil(idx / Lx) - 1)
    idy = idy - Int(ceil(idy / Ly) - 1)
    return getindex(envs.Envs, idx, idy)
end