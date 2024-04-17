function simple_update!(ipeps::iPEPSΓΛ, HamFunc::Function, para::Dict{Symbol,Any})
    Lx = ipeps.Lx
    Ly = ipeps.Ly

    τlis = para[:τlis]
    Dk = para[:Dk]
    verbose = para[:verbose]
    maxStep1τ = para[:maxStep1τ]
    hams = HamFunc(para)
    itsum = 0  # 记录总的迭代次数
    itime = 0.0  # 演化的总虚时间
    Ebefore = 0.0
    for τ in τlis
        gates = get_gates(hams, τ)
        for it in 0:maxStep1τ
            errlis, prodNrm = simple_update_1step!(ipeps, Dk, gates; verbose=verbose)
            itime += τ
            println("Truncation error = $(maximum(errlis)), total imaginary time = $itime")
            # ======== 检查能量收敛性 ====== See: PRB 104, 155118 (2021), Appendix 3.C
            E = -log(prodNrm) / τ
            println("Estimated energy per site is $(E / (Lx*Ly))")
            # =============================
            it += 1
            itsum += 1
            println("=========== Step τ=$τ, iteration $it, total iteration $itsum =======")
            println()
            # ======== 提前终止循环的情况 ===========
            if abs((E - Ebefore) / Ebefore) < para[:Etol]
                Ebefore = E
                println("!! Energy converge. Reduce imaginary time step")
                break
            end
            Ebefore = E
            flush(stdout)
        end
    end
    return nothing
end


"""
Generate trotter gates for simple update.\n
return a vector. [gateNN, gateNNN], etc.
"""
function get_gates(hams::Vector{TensorMap}, τ::Number)
    if length(hams) >= 3
        error("Only support up to next-nearest-neighbor interaction.")
    end
    gates = Vector{TensorMap}(undef, length(hams))
    gates[1] = exp(-τ * hams[1])
    # 注意次近邻哈密顿量要用两个对角路径求平均，用√gate
    length(hams) > 1 ? gates[2] = exp(-τ * hams[2] / 2) : nothing
    return gates
end


function simple_update_1step!(ipeps::iPEPSΓΛ, Dk::Int, gates::Vector{TensorMap}; verbose=1)
    Lx = ipeps.Lx
    Ly = ipeps.Ly
    errlis = Vector{Float64}(undef, 2 * length(gates) * Lx * Ly)  # 总的 bond 数
    prodNrm = 1.0
    Nb = 1
    # ================= 最近邻相互作用 ==============
    # 逐行更新横向Bond
    for yy in 1:Ly, xx in 1:Lx
        err, nrm = bond_proj_lr!(ipeps, xx, yy, Dk, gates[1])
        verbose > 1 ? println("横向更新xx$xx, yy$yy, error=$err") : nothing
        errlis[Nb] = err
        prodNrm *= nrm
        Nb += 1
    end
    # 逐列更新纵向Bond
    for xx in 1:Lx, yy in 1:Ly
        err, nrm = bond_proj_ud!(ipeps, xx, yy, Dk, gates[1])
        verbose > 1 ? println("纵向更新xx$xx, yy$yy, error=$err") : nothing
        errlis[Nb] = err
        prodNrm *= nrm
        Nb += 1
    end
    # ============= 次近邻相互作用 =============
    if length(gates) >= 2
        for yy in 1:Ly, xx in 1:Lx
            err, nrm = bond_proj_lu2rd!(ipeps, xx, yy, Dk, gates[2])
            verbose > 1 ? println("右下对角更新xx$xx, yy$yy, error=$err") : nothing
            errlis[Nb] = err
            prodNrm *= nrm
            Nb += 1

            err, nrm = bond_proj_ru2ld!(ipeps, xx, yy, Dk, gates[2])
            verbose > 1 ? println("左下对角更新xx$xx, yy$yy, error=$err") : nothing
            errlis[Nb] = err
            prodNrm *= nrm
            Nb += 1
        end
    end

    return errlis, prodNrm
end

# 一步 simple udpate
function simple_update_1step!(ipeps::iPEPSΓΛ, Dk::Int, gateNN::TensorMap; verbose=1)
    Lx = ipeps.Lx
    Ly = ipeps.Ly
    # ================= 最近邻相互作用 ==============
    errlis = Vector{Float64}(undef, 2 * Lx * Ly)  # 总的 bond 数
    prodNrm = 1.0
    Nb = 1
    # 逐行更新横向Bond
    for yy in 1:Ly, xx in 1:Lx
        err, nrm = bond_proj_lr!(ipeps, xx, yy, Dk, gateNN)
        verbose > 1 ? println("横向更新xx$xx, yy$yy, error=$err") : nothing
        errlis[Nb] = err
        prodNrm *= nrm
        Nb += 1
    end
    # 逐列更新纵向Bond
    for xx in 1:Lx, yy in 1:Ly
        err, nrm = bond_proj_ud!(ipeps, xx, yy, Dk, gateNN)
        verbose > 1 ? println("纵向更新xx$xx, yy$yy, error=$err") : nothing
        errlis[Nb] = err
        prodNrm *= nrm
        Nb += 1
    end
    # TODO ============= 次近邻相互作用 =============

    return errlis, prodNrm
end


# 更新 [xx, yy] 与 [xx+1, yy] 之间的 bond. See: SciPost Phys. Lect.Notes 25(2021)
function bond_proj_lr!(ipeps::iPEPSΓΛ, xx::Int, yy::Int, Dk::Int, gateNN::TensorMap)
    @tensor Γl[l, t, p; r, b] := ipeps[xx, yy].Γ[le, te, p, r, be] * ipeps[xx, yy].l[l, le] *
                                 ipeps[xx, yy].t[t, te] * ipeps[xx, yy].b[be, b]
    @tensor Γr[l, t, p; r, b] := ipeps[xx+1, yy].Γ[l, te, p, re, be] * ipeps[xx+1, yy].t[t, te] *
                                 ipeps[xx+1, yy].r[re, r] * ipeps[xx+1, yy].b[be, b]
    # 两次QR
    Xl, vl = leftorth(Γl, ((1, 2, 5), (3, 4)))
    wr, Yr = rightorth(Γr, ((1, 3), (2, 4, 5)))
    # 作用 Trotter Gate
    @tensor mid[(toX, pl, pr); (toY)] := vl[toX, plin, toV] * ipeps[xx, yy].r[toV, toW] * gateNN[pl, pr, plin, prin] *
                                         wr[toW, prin, toY]
    vnew, λnew, wnew, err = tsvd(mid, (1, 2), (3, 4); trunc=truncdim(Dk))
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

"""
更新 [xx, yy] 与 [xx+1, yy+1] 之间的 bond. (次近邻相互作用) See: PRB 82,245119(2010) \n
第一种路径，经过上方 site [xx+1, yy].  \n
Debug hint: [xx, yy] -> 1, [xx+1, yy] -> 2, [xx+1, yy+1] -> 3
"""
function site_proj_lu2rd_upPath!(ipeps::iPEPSΓΛ, xx::Int, yy::Int, Dk::Int, gateNNN::TensorMap)
    # gate和三个点上的张量都捏起来
    @tensor Θ[t1, l1, pu, b1; t2, r2, r3, b3, pd, l3, pmid] :=
        gateNNN[pu, pd, puin, pdin] *
        (ipeps[xx, yy].Γ[le1, te1, puin, re1, be1] * ipeps[xx, yy].l[l1, le1] * ipeps[xx, yy].t[t1, te1] * ipeps[xx, yy].r[re1, r1] * ipeps[xx, yy].b[be1, b1]) *
        (ipeps[xx+1, yy].Γ[r1, te2, pmid, re2, be2] * ipeps[xx+1, yy].t[t2, te2] * ipeps[xx+1, yy].r[re2, r2] * ipeps[xx+1, yy].b[be2, b2]) *
        (ipeps[xx+1, yy+1].Γ[le3, b2, pdin, re3, be3] * ipeps[xx+1, yy+1].l[l3, le3] * ipeps[xx+1, yy+1].r[re3, r3] * ipeps[xx+1, yy+1].b[be3, b3])
    # 分出 [xx, yy] 点，并做截断和归一
    Γ1p, λ1p, Θp, err1 = tsvd(Θ, ((1, 2, 3, 4), (5, 6, 7, 8, 9, 10, 11)); trunc=truncdim(Dk))
    nrm1 = norm(λ1p)
    λ1p = λ1p / nrm1
    # 分出 [xx+1, yy] 点，并做截断和归一
    @tensor Θpp[r2, t2, l2, pmid; r3, b3, pd, l3] := λ1p[l2, l2in] * Θp[l2in, t2, r2, r3, b3, pd, l3, pmid]
    Γ2p, λ2p, Γ3p, err2 = tsvd(Θpp, ((1, 2, 3, 4), (5, 6, 7, 8)); trunc=truncdim(Dk))
    nrm2 = norm(λ2p)
    λ2p = λ2p / nrm2
    # 恢复原来的张量及其指标顺序
    @tensor Γ1new[l1, t1, pu; r1, b1] := Γ1p[t1in, l1in, pu, b1in, r1] * inv(ipeps[xx, yy].l)[l1, l1in] * inv(ipeps[xx, yy].t)[t1, t1in] * inv(ipeps[xx, yy].b)[b1in, b1]
    @tensor Γ2new[l2, t2, pmid; r2, b2] := Γ2p[r2in, t2in, l2, pmid, b2] * inv(ipeps[xx+1, yy].t)[t2, t2in] * inv(ipeps[xx+1, yy].r)[r2in, r2]
    @tensor Γ3new[l3, t3, pd; r3, b3] := Γ3p[t3, r3in, b3in, pd, l3in] * inv(ipeps[xx+1, yy+1].l)[l3, l3in] * inv(ipeps[xx+1, yy+1].r)[r3in, r3] * inv(ipeps[xx+1, yy+1].b)[b3in, b3]
    # 更新 tensor
    ipeps[xx, yy].Γ = Γ1new
    ipeps[xx+1, yy].Γ = Γ2new
    ipeps[xx+1, yy+1].Γ = Γ3new
    ipeps[xx, yy].r = λ1p
    ipeps[xx+1, yy].l = λ1p
    ipeps[xx+1, yy].b = λ2p
    ipeps[xx+1, yy+1].t = λ2p

    return err1, err2, nrm1, nrm2
end

"""
更新 [xx, yy] 与 [xx+1, yy+1] 之间的 bond. (次近邻相互作用) See: PRB 82,245119(2010) \n
第二种路径，经过下方 site [xx, yy+1].  \n
Debug hint: [xx, yy] -> 1, [xx, yy+1] -> 2, [xx+1, yy+1] -> 3
"""
function site_proj_lu2rd_dnPath!(ipeps::iPEPSΓΛ, xx::Int, yy::Int, Dk::Int, gateNNN::TensorMap)
    # gate和三个点上的张量都捏起来
    @tensor Θ[r1, t1, l1, pu; t3, r3, b3, pd, b2, pmid, l2] :=
        gateNNN[pu, pd, puin, pdin] *
        (ipeps[xx, yy].Γ[le1, te1, puin, re1, be1] * ipeps[xx, yy].l[l1, le1] * ipeps[xx, yy].t[t1, te1] * ipeps[xx, yy].r[re1, r1] * ipeps[xx, yy].b[be1, b1]) *
        (ipeps[xx, yy+1].Γ[le2, b1, pmid, re2, be2] * ipeps[xx, yy+1].l[l2, le2] * ipeps[xx, yy+1].r[re2, r2] * ipeps[xx, yy+1].b[be2, b2]) *
        (ipeps[xx+1, yy+1].Γ[r2, te3, pdin, re3, be3] * ipeps[xx+1, yy+1].t[t3, te3] * ipeps[xx+1, yy+1].r[re3, r3] * ipeps[xx+1, yy+1].b[be3, b3])
    # 分出 [xx, yy] 点，并做截断和归一
    Γ1p, λ1p, Θp, err1 = tsvd(Θ, ((1, 2, 3, 4), (5, 6, 7, 8, 9, 10, 11)); trunc=truncdim(Dk))
    nrm1 = norm(λ1p)
    λ1p = λ1p / nrm1
    # 分出 [xx+1, yy] 点，并做截断和归一
    @tensor Θpp[t2, l2, pmid, b2; t3, r3, b3, pd] := λ1p[t2, t2in] * Θp[t2in, t3, r3, b3, pd, b2, pmid, l2]
    Γ2p, λ2p, Γ3p, err2 = tsvd(Θpp, ((1, 2, 3, 4), (5, 6, 7, 8)); trunc=truncdim(Dk))
    nrm2 = norm(λ2p)
    λ2p = λ2p / nrm2
    # 恢复原来的张量及其指标顺序
    @tensor Γ1new[l1, t1, pu; r1, b1] := Γ1p[r1in, t1in, l1in, pu, b1] * inv(ipeps[xx, yy].l)[l1, l1in] * inv(ipeps[xx, yy].t)[t1, t1in] * inv(ipeps[xx, yy].r)[r1in, r1]
    @tensor Γ2new[l2, t2, pmid; r2, b2] := Γ2p[t2, l2in, pmid, b2in, r2] * inv(ipeps[xx, yy+1].l)[l2, l2in] * inv(ipeps[xx, yy+1].b)[b2in, b2]
    @tensor Γ3new[l3, t3, pd; r3, b3] := Γ3p[l3, t3in, r3in, b3in, pd] * inv(ipeps[xx+1, yy+1].t)[t3, t3in] * inv(ipeps[xx+1, yy+1].r)[r3in, r3] * inv(ipeps[xx+1, yy+1].b)[b3in, b3]
    # 更新 tensor
    ipeps[xx, yy].Γ = Γ1new
    ipeps[xx, yy+1].Γ = Γ2new
    ipeps[xx+1, yy+1].Γ = Γ3new
    ipeps[xx, yy].b = λ1p
    ipeps[xx, yy+1].t = λ1p
    ipeps[xx, yy+1].r = λ2p
    ipeps[xx+1, yy+1].l = λ2p

    return err1, err2, nrm1, nrm2
end

# TODO here
"""
更新 [xx, yy] 与 [xx-1, yy+1] 之间的 bond. (次近邻相互作用) See: PRB 82,245119(2010) \n
第一种路径，经过上方 site [xx-1, yy].  \n
Debug hint: [xx, yy] -> 1, [xx-1, yy] -> 2, [xx-1, yy+1] -> 3
"""
function site_proj_ru2ld_upPath!(ipeps::iPEPSΓΛ, xx::Int, yy::Int, Dk::Int, gateNNN::TensorMap)
    # gate和三个点上的张量都捏起来
    @tensor Θ[t1, l1, pu, b1; t2, r2, r3, b3, pd, l3, pmid] :=
        gateNNN[pu, pd, puin, pdin] *
        (ipeps[xx, yy].Γ[le1, te1, puin, re1, be1] * ipeps[xx, yy].l[l1, le1] * ipeps[xx, yy].t[t1, te1] * ipeps[xx, yy].r[re1, r1] * ipeps[xx, yy].b[be1, b1]) *
        (ipeps[xx+1, yy].Γ[r1, te2, pmid, re2, be2] * ipeps[xx+1, yy].t[t2, te2] * ipeps[xx+1, yy].r[re2, r2] * ipeps[xx+1, yy].b[be2, b2]) *
        (ipeps[xx+1, yy+1].Γ[le3, b2, pdin, re3, be3] * ipeps[xx+1, yy+1].l[l3, le3] * ipeps[xx+1, yy+1].r[re3, r3] * ipeps[xx+1, yy+1].b[be3, b3])
    # 分出 [xx, yy] 点，并做截断和归一
    Γ1p, λ1p, Θp, err1 = tsvd(Θ, ((1, 2, 3, 4), (5, 6, 7, 8, 9, 10, 11)); trunc=truncdim(Dk))
    nrm1 = norm(λ1p)
    λ1p = λ1p / nrm1
    # 分出 [xx+1, yy] 点，并做截断和归一
    @tensor Θpp[r2, t2, l2, pmid; r3, b3, pd, l3] := λ1p[l2, l2in] * Θp[l2in, t2, r2, r3, b3, pd, l3, pmid]
    Γ2p, λ2p, Γ3p, err2 = tsvd(Θpp, ((1, 2, 3, 4), (5, 6, 7, 8)); trunc=truncdim(Dk))
    nrm2 = norm(λ2p)
    λ2p = λ2p / nrm2
    # 恢复原来的张量及其指标顺序
    @tensor Γ1new[l1, t1, pu; r1, b1] := Γ1p[t1in, l1in, pu, b1in, r1] * inv(ipeps[xx, yy].l)[l1, l1in] * inv(ipeps[xx, yy].t)[t1, t1in] * inv(ipeps[xx, yy].b)[b1in, b1]
    @tensor Γ2new[l2, t2, pmid; r2, b2] := Γ2p[r2in, t2in, l2, pmid, b2] * inv(ipeps[xx+1, yy].t)[t2, t2in] * inv(ipeps[xx+1, yy].r)[r2in, r2]
    @tensor Γ3new[l3, t3, pd; r3, b3] := Γ3p[t3, r3in, b3in, pd, l3in] * inv(ipeps[xx+1, yy+1].l)[l3, l3in] * inv(ipeps[xx+1, yy+1].r)[r3in, r3] * inv(ipeps[xx+1, yy+1].b)[b3in, b3]
    # 更新 tensor
    ipeps[xx, yy].Γ = Γ1new
    ipeps[xx+1, yy].Γ = Γ2new
    ipeps[xx+1, yy+1].Γ = Γ3new
    ipeps[xx, yy].r = λ1p
    ipeps[xx+1, yy].l = λ1p
    ipeps[xx+1, yy].b = λ2p
    ipeps[xx+1, yy+1].t = λ2p

    return err1, err2, nrm1, nrm2
end