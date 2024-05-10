"""
return the full environment of bond tensor at site1 and site2.
(The two sites should be nearest-neighbor now.)

`X` is the left (top) isometry while `Y` is the right (bottom) isometry.
"""
function get_Efull_fix_gauge(X::TensorMap, Xbar::TensorMap, Y::TensorMap, Ybar::TensorMap, envs::iPEPSenv, site1::Vector{Int}, site2::Vector{Int})
    # space(X) = [l, t, b; toV],  space(Y) = [toW; t, r, b]
    x1, y1 = site1
    x2, y2 = site2
    if x2 == x1 + 1
        gatel1 = swap_gate(space(X)[1], space(Xbar)[2]; Eltype=eltype(X))
        gatel2 = swap_gate(space(X)[3], space(Xbar)[4]; Eltype=eltype(X))
        gater1 = swap_gate(space(Y)[1], space(Ybar)[2]; Eltype=eltype(X))
        gater2 = swap_gate(space(Y)[4], space(Ybar)[3]; Eltype=eltype(X))
        @tensor Ql[rχt, ElDup; ElDdn, rχb] :=
            envs[x1, y1].corner.lt[rχin, bχin] * envs[x1, y1].transfer.t[rχin, rχt, tupDin, tdnDin] *
            envs[x1, y1].transfer.l[bχin, bχinin, lupDin, ldnDin] * gatel1[lupDin, tdnDin, lupDin2, tdnDin2] *
            X[lupDin2, tupDin, bupDin2, ElDup] * Xbar[ldnDin, tdnDin2, bdnDin, ElDdnin] *
            gatel2[bupDin, ElDdn, bupDin2, ElDdnin] * envs[x1, y1].corner.lb[bχinin, rχinin] *
            envs[x1, y1].transfer.b[rχinin, bupDin, bdnDin, rχb]
        @tensor Qr[lχt, ErDup; ErDdn, lχb] :=
            envs[x2, y2].corner.rt[lχin, bχin] * envs[x2, y2].transfer.t[lχt, lχin, tupDin, tdnDin] *
            envs[x2, y2].transfer.r[rupDin, rdnDin, bχin, bχinin] * Y[ErDupin, tupDin, rupDin, bupDin2] *
            gater1[ErDup, tdnDin, ErDupin, tdnDin2] * gater2[bupDin, rdnDin, bupDin2, rdnDin2] *
            Ybar[ErDdn, tdnDin2, rdnDin2, bdnDin] * envs[x2, y2].corner.rb[lχinin, bχinin] *
            envs[x2, y2].transfer.b[lχb, bupDin, bdnDin, lχinin]
        @tensor Efull[ElDup, ErDup; ElDdn, ErDdn] := Ql[rχt, ElDup; ElDdn, rχb] * Qr[rχt, ErDup, ErDdn, rχb]
        Efull, L, R, Linv, Rinv = fix_local_gauge!(Efull)
        return Efull, L, R, Linv, Rinv
    elseif y2 == y1 + 1
        gatet1 = swap_gate(space(X)[1], space(Xbar)[2]; Eltype=eltype(X))
        gatet2 = swap_gate(space(X)[3], space(Xbar)[4]; Eltype=eltype(X))
        gateb1 = swap_gate(space(Y)[1], space(Ybar)[2]; Eltype=eltype(X))
        gateb2 = swap_gate(space(Y)[4], space(Ybar)[3]; Eltype=eltype(X))
        @tensor Qt[bχl, EtDup; EtDdn, bχr] :=
            envs[x1, y1].corner.lt[rχin, bχin] * envs[x1, y1].transfer.t[rχin, rχinin, tupDin, tdnDin] *
            envs[x1, y1].transfer.l[bχin, bχl, lupDin, ldnDin] * gatet1[lupDin, tdnDin, lupDin2, tdnDin2] *
            X[lupDin2, tupDin, rupDin2, EtDup] * Xbar[ldnDin, tdnDin2, rdnDin, EtDdnin] *
            gatet2[rupDin, EtDdn, rupDin2, EtDdnin] * envs[x1, y1].corner.rt[rχinin, bχinin] *
            envs[x1, y1].transfer.r[rupDin, rdnDin, bχinin, bχr]
        @tensor Qb[tχl, EbDup; EbDdn, tχr] :=
            envs[x2, y2].corner.lb[tχin, rχin] * envs[x2, y2].transfer.b[rχin, bupDin, bdnDin, rχinin] *
            envs[x2, y2].transfer.l[tχl, tχin, lupDin, ldnDin] * Ybar[EbDdn, ldnDin2, rdnDin2, bdnDin] *
            gateb1[EbDup, ldnDin, EbDupin, ldnDin2] * gateb2[bupDin, rdnDin, bupDin2, rdnDin2] *
            Y[EbDupin, lupDin, rupDin, bupDin2] * envs[x2, y2].corner.rb[rχinin, tχinin] *
            envs[x2, y2].transfer.r[rupDin, rdnDin, tχr, tχinin]
        @tensor Efull[EtDup, EbDup; EtDdn, EbDdn] := Qt[bχl, EtDup; EtDdn, bχr] * Qb[bχl, EbDup; EbDdn, bχr]
        Efull, T, B, Tinv, Binv = fix_local_gauge!(Efull)
        return Efull, T, B, Tinv, Binv
    else
        error("check input sites")
    end
end


function fix_local_gauge!(Efull::TensorMap)
    # 左右/上下bond更新都直接用这个函数, 只要把index理解为 l->t, r->b 即可
    ε, W = eigh(Efull)
    ε = sqrt4diag(neg2zero(ε))  # √ε₊
    @tensor Z[ElDup, ErDup; mid] := W[ElDup, ErDup, in] * ε[in, mid]
    _, R = leftorth(Z, ((1, 3), (2,)))
    L, _ = rightorth(Z, ((1,), (2, 3)))
    Linv = inv(L)
    Rinv = inv(R)
    @tensor Efull[ElDup, ErDup; ElDdn, ErDdn] =
        Efull[ElDupin, ErDupin; ElDdnin, ErDdnin] * Linv[ElDup, ElDupin] * Linv'[ElDdnin, ElDdn] *
        Rinv[ErDupin, ErDup] * Rinv'[ErDdn, ErDdnin]
    # @assert ishermitian(Efull) "Efull should be Hermitian"
    return Efull, L, R, Linv, Rinv
end

"""
    `Ebarfull`: 环境 (fix local-gauge) 
    `L, R`: gauge tensor
    `vl, wr`: 初始的 bond tensor, 不带 guage
    `gateNN`: 两体哈密顿量
    return: `vlnew, wrnew` 更新后的bond tensor, 带着 gauge
    (See: SciPost Phys. Lect.Notes 25(2021) & PRB 92,035142(2015))
"""
function FFU_update_bond_lr(Ebarfull::T4, L::T2, R::T2, vl::T3v, wr::T3w, gateNN::T4, Dk::Int; tol, maxiter, verbose) where {T2,T3v,T3w,T4<:AbstractTensorMap}
    cost = NaN
    it = 0
    d0 = dold = zero(scalartype(vl))
    # 作用 Trotter Gate, 下面这一行还没有加gauge
    @tensor ṽw̃[ElDup, pl; pr, ErDup] := vl[ElDup, plin, mid] * gateNN[pl, pr, plin, prin] * wr[mid, prin, ErDup]
    # 用简单的 SVD 初始化vp, wp, 加入local gauge
    vp, S, wp, _ = tsvd(ṽw̃, ((1, 2), (3, 4)), trunc=truncdim(Dk))
    nrm = norm(S)
    S = S / nrm
    @tensor vp[(ElDup); (pl, mid)] = vp[ElDupin, pl, Sl] * sqrt4diag(S)[Sl, mid] * L[ElDupin, ElDup]
    @tensor wp[(mid); (pr, ErDup)] = sqrt4diag(S)[mid, Sr] * wp[Sr, pr, ErDupin] * R[ErDup, ErDupin]
    # 损失函数的通用部分, 加入 local gauge. 这里覆盖了 ṽw̃
    @tensor ṽw̃[ElDup, pl; pr, ErDup] = ṽw̃[ElDupin, pl, pr, ErDupin] * L[ElDupin, ElDup] * R[ErDup, ErDupin]
    @tensor Eṽw̃[pl, ElDdn; ErDdn, pr] := ṽw̃[ElDup, pl, pr, ErDup] * Ebarfull[ElDup, ErDup, ElDdn, ErDdn]
    @tensor Eṽdagw̃dag[pr, ErDup; ElDup, pl] := Ebarfull[ElDup, ErDup, ElDdn, ErDdn] * ṽw̃'[pr, ErDdn, ElDdn, pl]
    # 迭代更新vp, wp
    while it <= maxiter
        it += 1
        # 固定 wp 更新 vp. !! TensorMap 求 inv 后会交换 domain 和 codomain. !!
        @tensor S4v[pl, ElDdn, Srb] := Eṽw̃[pl, ElDdn, ErDdn, pr] * wp'[pr, ErDdn, Srb]
        @tensor R4v[ElDup, Srt; ElDdn, Srb] := Ebarfull[ElDup, ErDup, ElDdn, ErDdn] * wp'[pr, ErDdn, Srb] * wp[Srt, pr, ErDup]
        @tensor vptmp[(ElDup); (pl, Srt)] := inv(R4v)[ElDdn, Srb, ElDup, Srt] * S4v[pl, ElDdn, Srb]
        vp = vptmp
        # 固定 vp 更新 wp
        @tensor S4w[pr, ErDdn, Srb] := Eṽw̃[pl, ElDdn, ErDdn, pr] * vp'[pl, Srb, ElDdn]
        @tensor R4w[ErDup, Srt; ErDdn, Srb] := Ebarfull[ElDup, ErDup, ElDdn, ErDdn] * vp'[pl, Srb, ElDdn] * vp[ElDup, pl, Srt]
        @tensor wptmp[(Srt); (pr, ErDup)] := inv(R4w)[ErDdn, Srb, ErDup, Srt] * S4w[pr, ErDdn, Srb]
        wp = wptmp
        # 计算损失函数, ψp -> ψ' and ψt -> ψ̃
        @tensor ψpψp[] := vp[ElDup, pl, Srt] * wp[Srt, pr, ErDup] * Ebarfull[ElDup, ErDup, ElDdn, ErDdn] * vp'[pl, Srb, ElDdn] * wp'[pr, ErDdn, Srb]
        @tensor ψtψt[] := Eṽw̃[pl, ElDdn, ErDdn, pr] * ṽw̃'[pr, ErDdn, ElDdn, pl]
        @tensor ψtψp[] := Eṽw̃[pl, ElDdn, ErDdn, pr] * vp'[pl, Srb, ElDdn] * wp'[pr, ErDdn, Srb]
        @tensor ψpψt[] := Eṽdagw̃dag[pr, ErDup; ElDup, pl] * vp[ElDup, pl, Srt] * wp[Srt, pr, ErDup]
        dnew = scalar(ψpψp) + scalar(ψtψt) - scalar(ψtψp) - scalar(ψpψt)
        if it == 1
            d0 = dnew
            verbose > 1 ? println("FFU iteration $it. Tolerence $tol") : nothing
        else
            cost = abs(dnew - dold) / d0
            verbose > 1 ? println("FFU iteration $it, Cost function $cost.") : nothing
            cost < tol ? break : nothing
        end
        dold = dnew
    end
    return vp, wp, nrm, cost, it  # 这两个是带着 gauge 的
end


function FFU_update_bond_tb(Ebarfull::T4, T::T2, B::T2, vt::T3v, wb::T3w, gateNN::T4, Dk::Int; tol, maxiter, verbose) where {T2,T3v,T3w,T4<:AbstractTensorMap}
    cost = NaN
    it = 0
    d0 = dold = zero(scalartype(vt))
    # 作用 Trotter Gate, 下面这一行还没有加gauge
    @tensor ṽw̃[EtDup, pt; pb, EbDup] := vt[EtDup, ptin, mid] * gateNN[pt, pb, ptin, pbin] * wb[mid, pbin, EbDup]
    # 用简单的 SVD 初始化vp, wp, 加入local gauge
    vp, S, wp, _ = tsvd(ṽw̃, ((1, 2), (3, 4)), trunc=truncdim(Dk))
    nrm = norm(S)
    S = S / nrm
    @tensor vp[(EtDup); (pt, mid)] = vp[EtDupin, pt, St] * sqrt4diag(S)[St, mid] * T[EtDupin, EtDup]
    @tensor wp[(mid); (pb, EbDup)] = sqrt4diag(S)[mid, Sb] * wp[Sb, pb, EbDupin] * B[EbDup, EbDupin]
    # 损失函数的通用部分, 加入 local gauge. 这里覆盖了 ṽw̃
    @tensor ṽw̃[EtDup, pt; pb, EbDup] = ṽw̃[EtDupin, pt, pb, EbDupin] * T[EtDupin, EtDup] * B[EbDup, EbDupin]
    @tensor Eṽw̃[pt, EtDdn; EbDdn, pb] := ṽw̃[EtDup, pt, pb, EbDup] * Ebarfull[EtDup, EbDup, EtDdn, EbDdn]
    @tensor Eṽdagw̃dag[pb, EbDup; EtDup, pt] := Ebarfull[EtDup, EbDup, EtDdn, EbDdn] * ṽw̃'[pb, EbDdn, EtDdn, pt]
    # 迭代更新vp, wp
    while it <= maxiter
        it += 1
        # 固定 wp 更新 vp. !! TensorMap 求 inv 后会交换 domain 和 codomain. !!
        @tensor S4v[pt, EtDdn, Srb] := Eṽw̃[pt, EtDdn, EbDdn, pb] * wp'[pb, EbDdn, Srb]
        @tensor R4v[EtDup, Srt; EtDdn, Srb] := Ebarfull[EtDup, EbDup, EtDdn, EbDdn] * wp'[pb, EbDdn, Srb] * wp[Srt, pb, EbDup]
        @tensor vptmp[(EtDup); (pt, Srt)] := inv(R4v)[EtDdn, Srb, EtDup, Srt] * S4v[pt, EtDdn, Srb]
        vp = vptmp
        # 固定 vp 更新 wp
        @tensor S4w[pb, EbDdn, Srb] := Eṽw̃[pt, EtDdn, EbDdn, pb] * vp'[pt, Srb, EtDdn]
        @tensor R4w[EbDup, Srt; EbDdn, Srb] := Ebarfull[EtDup, EbDup, EtDdn, EbDdn] * vp'[pt, Srb, EtDdn] * vp[EtDup, pt, Srt]
        @tensor wptmp[(Srt); (pb, EbDup)] := inv(R4w)[EbDdn, Srb, EbDup, Srt] * S4w[pb, EbDdn, Srb]
        wp = wptmp
        # 计算损失函数, ψp -> ψ' and ψt -> ψ̃
        @tensor ψpψp[] := vp[EtDup, pt, Srt] * wp[Srt, pb, EbDup] * Ebarfull[EtDup, EbDup, EtDdn, EbDdn] * vp'[pt, Srb, EtDdn] * wp'[pb, EbDdn, Srb]
        @tensor ψtψt[] := Eṽw̃[pt, EtDdn, EbDdn, pb] * ṽw̃'[pb, EbDdn, EtDdn, pt]
        @tensor ψtψp[] := Eṽw̃[pt, EtDdn, EbDdn, pb] * vp'[pt, Srb, EtDdn] * wp'[pb, EbDdn, Srb]
        @tensor ψpψt[] := Eṽdagw̃dag[pb, EbDup; EtDup, pt] * vp[EtDup, pt, Srt] * wp[Srt, pb, EbDup]
        dnew = scalar(ψpψp) + scalar(ψtψt) - scalar(ψtψp) - scalar(ψpψt)
        if it == 1
            d0 = dnew
            verbose > 1 ? println("FFU iteration $it. Tolerence $tol") : nothing
        else
            cost = abs(dnew - dold) / d0
            verbose > 1 ? println("FFU iteration $it, Cost function $cost.") : nothing
            cost < tol ? break : nothing
        end
        dold = dnew
    end
    return vp, wp, nrm, cost, it  # 这两个是带着 gauge 的
end

"""
    做完一次bond更新后, 随之更新一次环境. 直接调用CTMRG的函数
"""
function FFU_update_env_lr!(ipeps::iPEPS, envs::iPEPSenv, xx::Int, χ::Int)
    Lx = ipeps.Lx
    ipepsbar = bar(ipeps)  # 这里后续可以优化, 应该不需要每次全部都求共轭
    for ii in xx:(Lx+xx-1)
        error_List = update_env_left_2by2!(ipeps, ipepsbar, envs, ii, χ)
    end
    for ii in (Lx+xx-1):-1:xx
        error_List = update_env_right_2by2!(ipeps, ipepsbar, envs, ii, χ)
    end
    return nothing
end

function FFU_update_env_tb!(ipeps::iPEPS, envs::iPEPSenv, yy::Int, χ::Int)
    Ly = ipeps.Ly
    ipepsbar = bar(ipeps)  # 这里后续可以优化, 应该不需要每次全部都求共轭
    for ii in yy:(Ly+yy-1)
        error_List = update_env_top_2by2!(ipeps, ipepsbar, envs, ii, χ)
    end
    for ii in (Ly+yy-1):-1:yy
        error_List = update_env_bottom_2by2!(ipeps, ipepsbar, envs, ii, χ)
    end
    return nothing
end

# ================================= 最近邻 ==============================================================
"""
FFU: 更新 `[xx, yy]` 与 `[xx+1, yy]` 之间的 bond. (See: SciPost Phys. Lect.Notes 25(2021) AND PRB 92,035142(2015))
"""
function bond_proj_lr!(ipeps::iPEPS, envs::iPEPSenv, xx::Int, yy::Int, Dk::Int, χ::Int, gateNN::TensorMap; tol, maxiter, verbose)
    # 计入第一个交换门, 把物理指标换过来
    swgate1 = swap_gate(space(ipeps[xx, yy])[3], space(ipeps[xx, yy])[5]; Eltype=eltype(ipeps[xx, yy]))
    @tensor Γl[l, t, p; r, b] := ipeps[xx, yy][l, t, pin, r, bin] * swgate1[p, b, pin, bin]
    Γlbar = bar(ipeps[xx, yy])
    Γrbar = bar(ipeps[xx+1, yy])
    # 两次QR
    Xl, vl = leftorth(Γl, ((1, 2, 5), (3, 4)))
    Xlbar, _ = leftorth(Γlbar, ((1, 2, 5), (3, 4)))
    wr, Yr = rightorth(ipeps[xx+1, yy], ((1, 3), (2, 4, 5)))
    _, Yrbar = rightorth(Γrbar, ((1, 3), (2, 4, 5)))
    # 求局域的环境
    Ebarfull, L, R, Linv, Rinv = get_Efull_fix_gauge(Xl, Xlbar, Yr, Yrbar, envs, [xx, yy], [xx + 1, yy])
    # 迭代更新这个bond
    vlnew, wrnew, nrm, cost, it = FFU_update_bond_lr(Ebarfull, L, R, vl, wr, gateNN, Dk; tol=tol, maxiter=maxiter, verbose=verbose)
    verbose > 0 ? println("FFU left-right update bond $([xx, yy]). Cost function $cost at iteration $(it)/$(maxiter).") : nothing
    # 新的两个 iPEPS 张量, 注意要把 gauge 去掉, 并且加上左边的交换门
    swgate2 = swap_gate(space(vlnew)[2], space(Xl)[3]; Eltype=eltype(vlnew))
    @tensor TensorLnew[l, t, p; r, b] := vlnew[tov, pin, r] * Linv[tov, toLinv] * Xl[l, t, bin, toLinv] * swgate2[p, b, pin, bin]
    @tensor TensorRnew[l, t, p; r, b] := wrnew[l, p, tow] * Rinv[toRinv, tow] * Yr[toRinv, t, r, b]
    # 更新 iPEPS tensors
    ipeps[xx+1, yy] = TensorRnew
    ipeps[xx, yy] = TensorLnew
    # 更新环境
    FFU_update_env_lr!(ipeps, envs, xx, χ)

    return nrm
end

"""
更新 `[xx, yy]` 与 `[xx, yy+1]` 之间的 bond. See: SciPost Phys. Lect.Notes 25(2021)
"""
function bond_proj_tb!(ipeps::iPEPS, envs::iPEPSenv, xx::Int, yy::Int, Dk::Int, χ::Int, gateNN::TensorMap; tol, maxiter, verbose)
    # 计入第一个交换门, 把下侧物理指标换过来
    swgate1 = swap_gate(space(ipeps[xx, yy+1])[3], space(ipeps[xx, yy+1])[1]; Eltype=eltype(ipeps[xx, yy+1]))
    @tensor Γb[l, t, p; r, b] := ipeps[xx, yy+1][lin, t, pin, r, b] * swgate1[p, l, pin, lin]
    Γtbar = bar(ipeps[xx, yy])
    Γbbar = bar(ipeps[xx, yy+1])
    # 两次QR
    Xt, vt = leftorth(ipeps[xx, yy], ((1, 2, 4), (3, 5)))
    Xtbar, _ = leftorth(Γtbar, ((1, 2, 4), (3, 5)))
    wb, Yb = rightorth(Γb, ((2, 3), (1, 4, 5)))
    _, Ybbar = rightorth(Γbbar, ((2, 3), (1, 4, 5)))
    # 求环境
    Ebarfull, T, B, Tinv, Binv = get_Efull_fix_gauge(Xt, Xtbar, Yb, Ybbar, envs, [xx, yy], [xx, yy + 1])
    # 迭代更新这个bond
    vtnew, wbnew, nrm, cost, it = FFU_update_bond_tb(Ebarfull, T, B, vt, wb, gateNN, Dk; tol=tol, maxiter=maxiter, verbose=verbose)
    verbose > 0 ? println("FFU top-bottom update bond $([xx, yy]). Cost function $cost at iteration $(it)/$(maxiter).") : nothing
    # 新的两个 iPEPS 张量, 注意要把 gauge 去掉, 并且加上下边的交换门
    swgate2 = swap_gate(space(wbnew)[2], space(Yb)[2]; Eltype=eltype(wbnew))
    @tensor TensorTnew[l, t, p; r, b] := vtnew[tov, p, b] * Tinv[tov, toTinv] * Xt[l, t, r, toTinv]
    @tensor TensorBnew[l, t, p; r, b] := wbnew[t, pin, tow] * Binv[toBinv, tow] * Yb[toBinv, lin, r, b] * swgate2[p, l, pin, lin]
    # 更新 iPEPS tensors
    ipeps[xx, yy] = TensorTnew
    ipeps[xx, yy+1] = TensorBnew
    # 更新环境
    FFU_update_env_tb!(ipeps, envs, yy, χ)

    return nrm
end