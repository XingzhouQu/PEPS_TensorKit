function simple_update!(ipeps::iPEPSΓΛ, Dk::Int, τlis::Vector{Number})
    Nit = length(τlis)
    for (it, τ) in enumerate(τlis)
        println("============= Simple update iteration $it / $Nit ================")
        gates = gen_gate(τ)
        errlis = simple_update_1step!(ipeps, Dk, gates)
        println("imaginary time now = $τ, truncation error = $(maximum(errlis))")
        flush(stdout)
    end

    return nothing
end


# 一步投影
function simple_update_1step!(ipeps::iPEPSΓΛ, Dk::Int, gates::Vector{TensorMap})
    gateNN = gates[1]
    Lx = ipeps.Lx
    Ly = ipeps.Ly
    # ================= 最近邻相互作用 ==============
    errlis = Vector{Float64}(undef, 2 * Lx * Ly)
    Nb = 1
    # 逐行更新横向Bond
    for yy in 1:Ly, xx in 1:Lx
        err = bond_proj_lr!(ipeps, xx, yy, Dk, gateNN)
        errlis[Nb] = err
        Nb += 1
    end
    # 逐列更新纵向Bond
    for xx in 1:Lx, yy in 1:Ly
        err = bond_proj_ud!(ipeps, xx, yy, Dk, gateNN)
        errlis[Nb] = err
        Nb += 1
    end
    # TODO ============= 次近邻相互作用 =============

    return errlis
end


# 更新 [xx, yy] 与 [xx+1, yy] 之间的 bond
function bond_proj_lr!(ipeps::iPEPSΓΛ, xx::Int, yy::Int, Dk::Int, gateNN::TensorMap)

    @tensor Γl[l, t, p; r, b] := ipeps[xx, yy].Γ[le, te, p, r, be] * ipeps[xx, yy].l[l, le] * ipeps[xx, yy].t[t, te] *
                                 ipeps[xx, yy].b[be, b]
    @tensor Γr[l, t, p; r, b] := ipeps[xx+1, yy].Γ[l, te, p, re, be] * ipeps[xx+1, yy].t[t, te] * ipeps[xx+1, yy].r[re, r] * ipeps[xx+1, yy].b[be, b]
    # 两次QR
    Xl, vl = leftorth(Γl, ((1, 2, 5), (3, 4)))
    wr, Yr = rightorth(Γr, ((1, 3), (2, 4, 5)))
    # 作用 Trotter Gate
    @tensor mid[toX, pl, pr; toY] := vl[toX, plin, toV] * ipeps[xx, yy].r[toV, toW] * gateNN[pl, pr, plin, prin] *
                                     wr[toW, prin, toY]
    vnew, λnew, wnew, err = tsvd(mid, ((1, 2), (3, 4)); trunc=truncdim(Dk))
    @tensor Γlnew[l, t, p; r, b] := Xl[le, te, be, toX] * vnew[toX, p, r] *
                                    inv(ipeps[xx, yy].l)[l, le] * inv(ipeps[xx, yy].t)[t, te] * inv(ipeps[xx, yy].b)[be, b]
    @tensor Γrnew[l, t, p; r, b] := wnew[l, p, toY] * Yr[toY, te, re, be] * inv(ipeps[xx+1, yy].t)[t, te] *
                                    inv(ipeps[xx+1, yy].r)[re, r] * inv(ipeps[xx+1, yy].b)[be, b]
    # 更新 tensor
    ipeps[xx, yy].r = λnew
    ipeps[xx+1, yy].l = λnew
    ipeps[xx, yy].Γ = Γlnew
    ipeps[xx+1, yy].Γ = Γrnew
    return err
end

# 更新 [xx, yy] 与 [xx, yy+1] 之间的 bond
function bond_proj_ud!(ipeps::iPEPSΓΛ, xx::Int, yy::Int, Dk::Int, gateNN::TensorMap)

    @tensor Γu[l, t, p; r, b] := ipeps[xx, yy].Γ[le, te, p, re, b] * ipeps[xx, yy].t[t, te] * ipeps[xx, yy].r[re, r] * ipeps[xx, yy].l[l, le]
    @tensor Γd[l, t, p; r, b] := ipeps[xx, yy+1].Γ[le, t, p, re, be] * ipeps[xx, yy+1].l[l, le] * ipeps[xx, yy+1].r[re, r] *
                                 ipeps[xx, yy+1].b[be, b]
    # 两次QR
    Xu, vu = leftorth(Γu, ((1, 2, 4), (3, 5)))
    wd, Yd = rightorth(Γd, ((2, 3), (1, 4, 5)))
    # 作用 Trotter Gate
    @tensor mid[toX, pu, pd; toY] := vu[toX, puin, toV] * ipeps[xx, yy].b[toV, toW] * gateNN[pu, pd, puin, pdin] *
                                     wd[toW, pdin, toY]
    vnew, λnew, wnew, err = tsvd(mid, ((1, 2), (3, 4)); trunc=truncdim(Dk))
    @tensor Γdnew[l, t, p; r, b] := Yd[toY, le, re, be] * wnew[t, p, toY] *
                                    inv(ipeps[xx, yy+1].l)[l, le] * inv(ipeps[xx, yy+1].r)[re, r] * inv(ipeps[xx, yy+1].b)[be, b]
    @tensor Γunew[l, t, p; r, b] := vnew[toX, p, b] * Xu[le, te, re, toX] * inv(ipeps[xx, yy].t)[t, te] *
                                    inv(ipeps[xx, yy].r)[re, r] * inv(ipeps[xx, yy].l)[l, le]
    # 更新 tensor
    ipeps[xx, yy].b = λnew
    ipeps[xx, yy+1].t = λnew
    ipeps[xx, yy].Γ = Γunew
    ipeps[xx, yy+1].Γ = Γdnew
    return err
end