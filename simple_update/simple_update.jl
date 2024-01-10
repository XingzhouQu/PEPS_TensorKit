function simple_update!(ipeps::iPEPSΓΛ, Dk::Int, τlis::Vector{Number})
    Nit = length(τlis)
    for (it, τ) in enumerate(τlis)
        println("========= Simple update iteration $it / $Nit ===================")
        gates = gen_gate(τ)
        errlis = simple_update_1step!(ipeps, Dk, gates)
        println("imaginary time now = $τ, truncation error = $(maximum(errlis))")
        flush(stdout)
    end
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
    Γl = ipeps[xx, yy].Γ
    Γr = ipeps[xx+1, yy].Γ
    swgate = swap_gate(space(Γl)[3]', space(Γl)[5]')
    @tensor Γl[l, t, p; r, b] = Γl[le, te, pin, re, be] * ipeps[xx, yy].l[l, le] * ipeps[xx, yy].t[t, tin] *
                                ipeps[xx, yy].b[be, bin] * swgate[pin, bin, p, b]
    @tensor Γr[l, t, p; r, b] = Γr[l, te, p, re, be] * ipeps[xx+1, yy].t[t, te] * ipeps[xx+1, yy].r[re, r] * ipeps[xx+1, yy].b[be, b]
    Xl, vl = leftorth(Γl, ((1, 2, 5), (3, 4)))
    wr, Yr = rightorth(Γr, ((1, 3), (2, 4, 5)))
    @tensor mid[toX, pl, pr; toY] := vl[toX, plin, toV] * ipeps[xx, yy].r[toV, toW] * gateNN[pl, pr, plin, prin] *
                                     wr[toY, toW, prin]
    vnew, λnew, wnew, err = tsvd(mid, ((1, 2), (3, 4)); trunc=truncdim(Dk))


    return err
end

# 更新 [xx, yy] 与 [xx, yy+1] 之间的 bond
function bond_proj_ud!(ipeps::iPEPSΓΛ, xx::Int, yy::Int, Dk::Int, gateNN::TensorMap)


    return err
end