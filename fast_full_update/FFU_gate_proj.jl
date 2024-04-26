"""
return the full environment of bond tensor at site1 and site2.
(The two sites should be nearest-neighbor now.)

`X` is the left (top) isometry while `Y` is the right (bottom) isometry.
"""
function get_Efull_fix_gauge(X::TensorMap, Y::TensorMap, envs::iPEPSenv, site1::Vector{Int}, site2::Vector{Int})
    x1, y1 = site1
    x2, y2 = site2
    if x2 == x1 + 1
        @tensor Ql[rχt, ElDup; ElDdn, rχb] :=
            envs[x1, y1].corner.lt[rχin, bχin] * envs[x1, y1].transfer.t[rχin, rχt, tupDin, tdnDin] *  
            X[lupDin, tupDin, bupDin, ElDup] * X'[ElDdn, ldnDin, tdnDin, bdnDin] * 
            envs[x1, y1].transfer.l[bχin, bχinin, lupDin, ldnDin] * envs[x1, y1].corner.lb[bχinin, rχinin] * 
            envs[x1, y1].transfer.b[rχinin, bupDin, bdnDin, rχb]
        @tensor Qr[lχt, ErDup; ErDdn, lχb] :=
            envs[x2, y2].corner.rt[lχin, bχin] * envs[x2, y2].transfer.t[lχt, lχin, tupDin, tdnDin] *  
            Y[ErDup, tupDin, rupDin, bupDin] * Y'[tdnDin, rdnDin, bdnDin, ErDdn] * 
            envs[x2, y2].transfer.r[rupDin, rdnDin, bχin, bχinin] * envs[x2, y2].corner.rb[lχinin, bχinin] * 
            envs[x2, y2].transfer.b[lχb, bupDin, bdnDin, lχinin]
        @tensor Efull[ElDup, ErDup; ElDdn, ErDdn] := Ql[rχt, ElDup; ElDdn, rχb] * Qr[rχt, ErDup, ErDdn, rχb]
        Efull, L, R, Linv, Rinv = fix_local_gauge!(Efull)
    elseif y2 == y1 + 1
        
    else
        error("check input sites")
    end
    return Efull, L, R
end


function fix_local_gauge!(Efull::TensorMap)
    @assert ishermitian(Efull) "Efull should be Hermitian"
    ε, W = eigh(Efull)
    ε = sqrt4diag(neg2zero(ε))  # √ε₊
    @tensor Z[ElDup, ErDup; mid] := W[ElDup, ErDup, in] * ε[in, mid]
    _, R = leftorth(Z, ((1, 3), (2,)))
    L, _ = rightorth(Z, ((1, ), (2, 3)))
    Linv = inv(L)
    Rinv = inv(R)
    @tensor Efull[ElDup, ErDup; ElDdn, ErDdn] = Efull[ElDupin, ErDupin; ElDdnin, ErDdnin] * 
        Linv[ElDup, ElDupin] * Linv'[ElDdnin, ElDdn] * Rinv[ErDupin, ErDup] * Rinv'[ErDup, ErDdnin]
    return Efull, L, R, Linv, Rinv
end


# ================================= 最近邻 ==============================================================
"""
更新 `[xx, yy]` 与 `[xx+1, yy]` 之间的 bond. See: SciPost Phys. Lect.Notes 25(2021)
"""
function bond_proj_lr!(ipeps::iPEPS, envs::iPEPSenv, xx::Int, yy::Int, Dk::Int, gateNN::TensorMap; tol=)
    # 两次QR
    Xl, vl = leftorth(ipeps[xx, yy], ((1, 2, 5), (3, 4)))
    wr, Yr = rightorth(ipeps[xx+1, yy], ((1, 3), (2, 4, 5)))
    # 作用 Trotter Gate
    @tensor ṽw̃[(toX, pl, pr); (toY)] := vl[toX, plin, mid] * gateNN[pl, pr, plin, prin] * wr[mid, prin, toY]
    # 求环境
    Ebarfull, L, R = get_Efull_fix_gauge(Xl, Yr, envs, [xx, yy], [xx+1, yy])
    @tensor Eṽw̃[] := L[] * ṽw̃[] * R[] * Ebarfull[]

    vnew, λnew, wnew, err = tsvd(ṽw̃, (1, 2), (3, 4); trunc=truncdim(Dk))
    @tensor Γlnew[l, t, p; r, b] := Xl[le, te, be, toX] * vnew[toX, p, r] *
                                    inv(ipeps[xx, yy].l)[l, le] * inv(ipeps[xx, yy].t)[t, te] * inv(ipeps[xx, yy].b)[be, b]
    @tensor Γrnew[l, t, p; r, b] := wnew[l, p, toY] * Yr[toY, te, re, be] * inv(ipeps[xx+1, yy].t)[t, te] *
                                    inv(ipeps[xx+1, yy].r)[re, r] * inv(ipeps[xx+1, yy].b)[be, b]
    # 更新 tensor, 注意这里要归一化
    nrm = norm(λnew)
    λnew = λnew / nrm
    ipeps[xx+1, yy].Γ = Γrnew
    ipeps[xx, yy].Γ = Γlnew
    ipeps[xx+1, yy].l = λnew
    ipeps[xx, yy].r = λnew
    return err, nrm
end

"""
更新 `[xx, yy]` 与 `[xx, yy+1]` 之间的 bond. See: SciPost Phys. Lect.Notes 25(2021)
"""
function bond_proj_tb!(ipeps::iPEPS, envs::iPEPSenv, xx::Int, yy::Int, Dk::Int, gateNN::TensorMap)
    @tensor Γu[l, t, p; r, b] := ipeps[xx, yy].Γ[le, te, p, re, b] * ipeps[xx, yy].t[t, te] * ipeps[xx, yy].r[re, r] * ipeps[xx, yy].l[l, le]
    @tensor Γd[l, t, p; r, b] := ipeps[xx, yy+1].Γ[le, t, p, re, be] * ipeps[xx, yy+1].l[l, le] * ipeps[xx, yy+1].r[re, r] *
                                 ipeps[xx, yy+1].b[be, b]
    # 两次QR
    Xu, vu = leftorth(Γu, ((1, 2, 4), (3, 5)))
    wd, Yd = rightorth(Γd, ((2, 3), (1, 4, 5)))
    # 作用 Trotter Gate
    @tensor mid[toX, pu, pd; toY] := vu[toX, puin, toV] * ipeps[xx, yy].b[toV, toW] * gateNN[pu, pd, puin, pdin] *
                                     wd[toW, pdin, toY]
    vnew, λnew, wnew, err = tsvd(mid, (1, 2), (3, 4); trunc=truncdim(Dk))
    @tensor Γdnew[l, t, p; r, b] := Yd[toY, le, re, be] * wnew[t, p, toY] *
                                    inv(ipeps[xx, yy+1].l)[l, le] * inv(ipeps[xx, yy+1].r)[re, r] * inv(ipeps[xx, yy+1].b)[be, b]
    @tensor Γunew[l, t, p; r, b] := vnew[toX, p, b] * Xu[le, te, re, toX] * inv(ipeps[xx, yy].t)[t, te] *
                                    inv(ipeps[xx, yy].r)[re, r] * inv(ipeps[xx, yy].l)[l, le]
    # 更新 tensor, 注意这里要归一化
    nrm = norm(λnew)
    λnew = λnew / nrm
    ipeps[xx, yy].b = λnew
    ipeps[xx, yy+1].t = λnew
    ipeps[xx, yy].Γ = Γunew
    ipeps[xx, yy+1].Γ = Γdnew
    return err, nrm
end