# ================================= 最近邻 ==============================================================
"""
更新 `[xx, yy]` 与 `[xx+1, yy]` 之间的 bond. See: SciPost Phys. Lect.Notes 25(2021) and Master thesis of Chang-Kai Zhang
"""
function bond_proj_lr!(ipeps::iPEPSΓΛ, xx::Int, yy::Int, Dk::Int, gateNN::TensorMap)
    # 计入第一个交换门, 把物理指标换过来
    swgate1 = swap_gate(space(ipeps[xx, yy].Γ)[3], space(ipeps[xx, yy].Γ)[5]; Eltype=eltype(ipeps[xx, yy].Γ))
    @tensor Γltmp[l, t, p; r, b] := ipeps[xx, yy].Γ[l, t, pin, r, bin] * swgate1[p, b, pin, bin]
    # 开始玻色型的分解
    @tensor Γl[l, t, p; r, b] := Γltmp[le, te, p, r, be] * ipeps[xx, yy].l[l, le] *
                                 ipeps[xx, yy].t[t, te] * ipeps[xx, yy].b[be, b]
    @tensor Γr[l, t, p; r, b] := ipeps[xx+1, yy].Γ[l, te, p, re, be] * ipeps[xx+1, yy].t[t, te] *
                                 ipeps[xx+1, yy].r[re, r] * ipeps[xx+1, yy].b[be, b]
    # 两次QR
    Xl, vl = leftorth(Γl, ((1, 2, 5), (3, 4)))
    wr, Yr = rightorth(Γr, ((1, 3), (2, 4, 5)))
    # 作用 Trotter Gate
    @tensor mid[(toX, pl, pr); (toY)] := vl[toX, plin, toV] * ipeps[xx, yy].r[toV, toW] * gateNN[pl, pr, plin, prin] *
                                         wr[toW, prin, toY]
    vnew, λnew, wnew, err = tsvd(mid, (1, 2), (3, 4); trunc=truncdim(Dk), alg=TensorKit.SVD())
    # 计入第二个交换门，把物理指标换回去
    swgate2 = swap_gate(space(Xl)[3], space(vnew)[2]; Eltype=eltype(vnew))
    @tensor Γlnew[l, t, p; r, b] := Xl[le, te, bein, toX] * vnew[toX, pin, r] * swgate2[be, p, bein, pin] *
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
更新 `[xx, yy]` 与 `[xx, yy+1]` 之间的 bond. See: SciPost Phys. Lect.Notes 25(2021) and Master thesis of Chang-Kai Zhang
"""
function bond_proj_tb!(ipeps::iPEPSΓΛ, xx::Int, yy::Int, Dk::Int, gateNN::TensorMap)
    # 计入第一个交换门，把物理指标换过来
    swgate1 = swap_gate(space(ipeps[xx, yy+1].Γ)[1], space(ipeps[xx, yy+1].Γ)[3]; Eltype=eltype(ipeps[xx, yy+1].Γ))
    @tensor Γdtmp[l, t, p; r, b] := ipeps[xx, yy+1].Γ[lin, t, pin, r, b] * swgate1[l, p, lin, pin]
    # 开始玻色型的分解
    @tensor Γu[l, t, p; r, b] := ipeps[xx, yy].Γ[le, te, p, re, b] * ipeps[xx, yy].t[t, te] * ipeps[xx, yy].r[re, r] * ipeps[xx, yy].l[l, le]
    @tensor Γd[l, t, p; r, b] := Γdtmp[le, t, p, re, be] * ipeps[xx, yy+1].l[l, le] * ipeps[xx, yy+1].r[re, r] *
                                 ipeps[xx, yy+1].b[be, b]
    # 两次QR
    Xu, vu = leftorth(Γu, ((1, 2, 4), (3, 5)))
    wd, Yd = rightorth(Γd, ((2, 3), (1, 4, 5)))
    # 作用 Trotter Gate
    @tensor mid[toX, pu, pd; toY] := vu[toX, puin, toV] * ipeps[xx, yy].b[toV, toW] * gateNN[pu, pd, puin, pdin] *
                                     wd[toW, pdin, toY]
    vnew, λnew, wnew, err = tsvd(mid, (1, 2), (3, 4); trunc=truncdim(Dk), alg=TensorKit.SVD())
    # 计入第二个交换门，把物理指标换回去
    swgate2 = swap_gate(space(Yd)[2], space(wnew)[2]; Eltype=eltype(wnew))
    @tensor Γdnew[l, t, p; r, b] := Yd[toY, lein, re, be] * wnew[t, pin, toY] * swgate2[le, p, lein, pin] *
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

# ================================= 次近邻 ==============================================================

# 次近邻这部分，在做SVD的时候要注意 U 和 V 的位置（哪个是U，哪个是V），最简单的标准是让奇异值谱能直接作为 bond tensor而不用取dagger.
# debug hint: (1) sitelu   (2) auxsite      or      (1) sitelu    
#                          (3) siterd               (2) auxsite    (3) siterd
#
# debug hint: (2) auxsite  (1) siteru       or                     (1) siteru
#             (3) siteld                            (3) siteld     (2) auxsite
"""
更新 `[xx, yy]` 与 `[xx+1, yy+1]` 之间的 bond. (次近邻相互作用) See: PRB 82,245119(2010) \n
第一种路径，经过上方 site `[xx+1, yy]`.  \n
Debug hint: [xx, yy] -> 1, [xx+1, yy] -> 2, [xx+1, yy+1] -> 3
"""
function site_proj_lu2rd_upPath!(ipeps::iPEPSΓΛ, xx::Int, yy::Int, Dk::Int, gateNNN::TensorMap)
    # gate和三个点上的张量都捏起来, 3 site update 方法
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

function bond_proj_lu2rd_upPath!(ipeps::iPEPSΓΛ, xx::Int, yy::Int, Dk::Int, gateNNN::TensorMap)
    # 1 site update 方法
    # 先计入 sitelu, siterd 内部交换门
    swgt1 = swap_gate(space(ipeps[xx, yy].Γ)[3], space(ipeps[xx, yy].Γ)[5]; Eltype=eltype(ipeps[xx, yy].Γ))
    @tensor γ1[l, t, p; r, b] := swgt1[p, b, pin, bin] * ipeps[xx, yy].Γ[l, t, pin, r, bin]
    @tensor Γ1[l1, t1, pu; r1, b1] := γ1[le1, te1, pu, r1, be1] * ipeps[xx, yy].l[l1, le1] * ipeps[xx, yy].t[t1, te1] * ipeps[xx, yy].b[be1, b1]
    swgt3 = swap_gate(space(ipeps[xx+1, yy+1].Γ)[3], space(ipeps[xx+1, yy+1].Γ)[1]; Eltype=eltype(ipeps[xx, yy].Γ))
    @tensor γ3[l, t, p; r, b] := swgt3[p, l, pin, lin] * ipeps[xx+1, yy+1].Γ[lin, t, pin, r, b]
    @tensor Γ3[l3, t3, pd; r3, b3] := γ3[le3, t3, pd, re3, be3] * ipeps[xx+1, yy+1].l[l3, le3] * ipeps[xx+1, yy+1].r[re3, r3] * ipeps[xx+1, yy+1].b[be3, b3]
    # 两次QR
    X1, v1 = leftorth(Γ1, ((1, 2, 5), (3, 4)))
    v3, X3 = rightorth(Γ3, ((2, 3), (1, 4, 5)))
    # 两个交换门
    swgtmid1 = swap_gate(space(ipeps[xx+1, yy].Γ)[3], space(gateNNN)[4]; Eltype=eltype(v1))
    swgtmid2 = swap_gate(space(swgtmid1)[1], space(gateNNN)[2]; Eltype=eltype(v1))
    # 两个bond tensor 与 auxsite 收缩
    @tensor Θ[toΓ1, pu; t2, pmid, r2, pd, toΓ3] :=
        v1[toΓ1, puin, re1] * ipeps[xx, yy].r[re1, r1] * ipeps[xx+1, yy].Γ[r1, te2, pmidin, re2, be2] *
        ipeps[xx+1, yy].t[t2, te2] * ipeps[xx+1, yy].r[re2, r2] * ipeps[xx+1, yy].b[be2, b2] * v3[b2, pdin, toΓ3] *
        swgtmid1[pmidin2, pdin, pmidin, pdin2] * swgtmid2[pmid, pd, pmidin2, pd2] * gateNNN[pu, pd2, puin, pdin2]
    # 分出 [xx, yy] 点的 v1，并做截断和归一.
    v1new, λ1p, Θp, err1 = tsvd(Θ, ((1, 2), (3, 4, 5, 6, 7)); trunc=truncdim(Dk), alg=TensorKit.SVD())
    nrm1 = norm(λ1p)
    λ1p = λ1p / nrm1
    # 分出 [xx+1, yy] 点，并做截断和归一
    @tensor Θpp[l2, t2, pmid, r2; pd, toΓ3] := λ1p[l2, l2in] * Θp[l2in, t2, pmid, r2, pd, toΓ3]
    Γ2p, λ2p, v3new, err2 = tsvd(Θpp, ((1, 2, 3, 4), (5, 6)); trunc=truncdim(Dk), alg=TensorKit.SVD())
    nrm2 = norm(λ2p)
    λ2p = λ2p / nrm2
    # 恢复原来的张量及其指标顺序.
    swgtlu = swap_gate(space(v1new)[2], space(X1)[3]; Eltype=eltype(v1new))
    swgtrd = swap_gate(space(v3new)[2], space(X3)[2]; Eltype=eltype(v3new))
    @tensor Γ1new[l1, t1, pu; r1, b1] :=
        X1[l1in, t1in, b1inp, toΓ1] * v1new[toΓ1, puin, r1] * swgtlu[pu, b1in, puin, b1inp] *
        inv(ipeps[xx, yy].l)[l1, l1in] * inv(ipeps[xx, yy].t)[t1, t1in] * inv(ipeps[xx, yy].b)[b1in, b1]
    @tensor Γ2new[l2, t2, pmid; r2, b2] :=
        Γ2p[l2in, t2in, pmid, r2in, b2] * inv(ipeps[xx+1, yy].t)[t2, t2in] * inv(ipeps[xx+1, yy].r)[r2in, r2] *
        inv(λ1p)[l2, l2in]
    @tensor Γ3new[l3, t3, pd; r3, b3] :=
        v3new[t3, pdin, toΓ3] * X3[toΓ3, l3inp, r3in, b3in] * swgtrd[pd, l3in, pdin, l3inp] *
        inv(ipeps[xx+1, yy+1].l)[l3, l3in] * inv(ipeps[xx+1, yy+1].r)[r3in, r3] * inv(ipeps[xx+1, yy+1].b)[b3in, b3]
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
    # 分出 [xx, yy+1] 点，并做截断和归一
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

function bond_proj_lu2rd_dnPath!(ipeps::iPEPSΓΛ, xx::Int, yy::Int, Dk::Int, gateNNN::TensorMap)
    # 1 site update 方法
    # 先计入交换门
    swgt1 = swap_gate(space(ipeps[xx, yy].Γ)[3], space(ipeps[xx, yy].Γ)[5]; Eltype=eltype(ipeps[xx, yy].Γ))
    swgt2 = swap_gate(space(ipeps[xx+1, yy+1].Γ)[3], space(ipeps[xx+1, yy+1].Γ)[1]; Eltype=eltype(ipeps[xx, yy].Γ))
    @tensor γ1[l, t, p; r, b] := ipeps[xx, yy].Γ[l, t, pin, r, bin] * swgt1[p, b, pin, bin]
    @tensor Γ1[l1, t1, pu; r1, b1] := γ1[le1, te1, pu, re1, b1] * ipeps[xx, yy].l[l1, le1] * ipeps[xx, yy].t[t1, te1] * ipeps[xx, yy].r[re1, r1]
    @tensor γ3[l, t, p; r, b] := ipeps[xx+1, yy+1].Γ[lin, t, pin, r, b] * swgt2[p, l, pin, lin]
    @tensor Γ3[l3, t3, pd; r3, b3] := γ3[l3, te3, pd, re3, be3] * ipeps[xx+1, yy+1].t[t3, te3] * ipeps[xx+1, yy+1].r[re3, r3] * ipeps[xx+1, yy+1].b[be3, b3]
    # 两次QR
    X1, v1 = leftorth(Γ1, ((1, 2, 4), (3, 5)))
    v3, X3 = rightorth(Γ3, ((1, 3), (2, 4, 5)))
    # 两个交换门
    swgtmid1 = swap_gate(space(v1)[3], space(gateNNN)[1]; Eltype=eltype(v1))
    swgtmid2 = swap_gate(space(v3)[1], space(gateNNN)[2]; Eltype=eltype(v1))
    # 两个bond tensor 与 auxsite 收缩
    @tensor Θ[toΓ1, pu; l2, pmid, b2, pd, toΓ3] :=
        v1[toΓ1, puin, be1in] * swgtmid1[be1, pu, be1in, pup] * ipeps[xx, yy].b[be1, b1] *
        gateNNN[pup, pdp, puin, pdin] * ipeps[xx, yy+1].Γ[le2, b1, pmid, re2, be2] *
        ipeps[xx, yy+1].l[l2, le2] * ipeps[xx, yy+1].r[re2, r2] * ipeps[xx, yy+1].b[be2, b2] *
        swgtmid2[r2, pd, r2in, pdp] * v3[r2in, pdin, toΓ3]
    # 分出 [xx, yy] 点的 v1，并做截断和归一.
    v1new, λ1p, Θp, err1 = tsvd(Θ, ((1, 2), (3, 4, 5, 6, 7)); trunc=truncdim(Dk), alg=TensorKit.SVD())
    nrm1 = norm(λ1p)
    λ1p = λ1p / nrm1
    # 分出 [xx, yy+1] 点，并做截断和归一
    @tensor Θpp[l2, t2, pmid, b2; pd, toΓ3] := λ1p[t2, t2in] * Θp[t2in, l2, pmid, b2, pd, toΓ3]
    Γ2p, λ2p, v3new, err2 = tsvd(Θpp, ((1, 2, 3, 4), (5, 6)); trunc=truncdim(Dk), alg=TensorKit.SVD())
    nrm2 = norm(λ2p)
    λ2p = λ2p / nrm2
    # 恢复原来的张量及其指标顺序.
    @tensor Γ1new[l1, t1, pu; r1, b1] := X1[l1in, t1in, r1in, toΓ1] * v1new[toΓ1, pu, b1] * inv(ipeps[xx, yy].l)[l1, l1in] * inv(ipeps[xx, yy].t)[t1, t1in] * inv(ipeps[xx, yy].r)[r1in, r1]
    @tensor Γ2new[l2, t2, pmid; r2, b2] := Γ2p[l2in, t2in, pmid, b2in, r2] * inv(ipeps[xx, yy+1].l)[l2, l2in] * inv(ipeps[xx, yy+1].b)[b2in, b2] * inv(λ1p)[t2, t2in]
    @tensor Γ3new[l3, t3, pd; r3, b3] := v3new[l3, pd, toΓ3] * X3[toΓ3, t3in, r3in, b3in] * inv(ipeps[xx+1, yy+1].t)[t3, t3in] * inv(ipeps[xx+1, yy+1].r)[r3in, r3] * inv(ipeps[xx+1, yy+1].b)[b3in, b3]
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


"""
更新 [xx, yy] 与 [xx-1, yy+1] 之间的 bond. (次近邻相互作用) See: PRB 82,245119(2010) \n
第一种路径，经过上方 site [xx-1, yy].  \n
Debug hint: [xx, yy] -> 1, [xx-1, yy] -> 2, [xx-1, yy+1] -> 3
"""
function site_proj_ru2ld_upPath!(ipeps::iPEPSΓΛ, xx::Int, yy::Int, Dk::Int, gateNNN::TensorMap)
    # gate和三个点上的张量都捏起来
    @tensor contractcheck = true Θ[l2, t2, pmid, l3, pd, r3, b3; t1, pu, r1, b1] :=
        gateNNN[pu, pd, puin, pdin] *
        (ipeps[xx, yy].Γ[le1, te1, puin, re1, be1] * ipeps[xx, yy].l[l1, le1] * ipeps[xx, yy].t[t1, te1] * ipeps[xx, yy].r[re1, r1] * ipeps[xx, yy].b[be1, b1]) *
        (ipeps[xx-1, yy].Γ[le2, te2, pmid, l1, be2] * ipeps[xx-1, yy].t[t2, te2] * ipeps[xx-1, yy].l[l2, le2] * ipeps[xx-1, yy].b[be2, b2]) *
        (ipeps[xx-1, yy+1].Γ[le3, b2, pdin, re3, be3] * ipeps[xx-1, yy+1].l[l3, le3] * ipeps[xx-1, yy+1].r[re3, r3] * ipeps[xx-1, yy+1].b[be3, b3])
    # 分出 [xx, yy] 点，并做截断和归一
    Θp, λ1p, Γ1p, err1 = tsvd(Θ, ((1, 2, 3, 4, 5, 6, 7), (8, 9, 10, 11)); trunc=truncdim(Dk))
    nrm1 = norm(λ1p)
    λ1p = λ1p / nrm1
    # 分出 [xx-1, yy] 点，并做截断和归一
    @tensor Θpp[l2, t2, pmid, r2; l3, pd, r3, b3] := λ1p[r2in, r2] * Θp[l2, t2, pmid, l3, pd, r3, b3, r2in]
    Γ2p, λ2p, Γ3p, err2 = tsvd(Θpp, ((1, 2, 3, 4), (5, 6, 7, 8)); trunc=truncdim(Dk))
    nrm2 = norm(λ2p)
    λ2p = λ2p / nrm2
    # 恢复原来的张量及其指标顺序
    @tensor Γ1new[l1, t1, pu; r1, b1] := Γ1p[l1, t1in, pu, r1in, b1in] * inv(ipeps[xx, yy].r)[r1in, r1] * inv(ipeps[xx, yy].t)[t1, t1in] * inv(ipeps[xx, yy].b)[b1in, b1]
    @tensor Γ2new[l2, t2, pmid; r2, b2] := Γ2p[l2in, t2in, pmid, r2, b2] * inv(ipeps[xx-1, yy].t)[t2, t2in] * inv(ipeps[xx-1, yy].l)[l2, l2in]
    @tensor Γ3new[l3, t3, pd; r3, b3] := Γ3p[t3, l3in, pd, r3in, b3in] * inv(ipeps[xx-1, yy+1].l)[l3, l3in] * inv(ipeps[xx-1, yy+1].r)[r3in, r3] * inv(ipeps[xx-1, yy+1].b)[b3in, b3]
    # 更新 tensor
    ipeps[xx, yy].Γ = Γ1new
    ipeps[xx-1, yy].Γ = Γ2new
    ipeps[xx-1, yy+1].Γ = Γ3new
    ipeps[xx, yy].l = λ1p
    ipeps[xx-1, yy].r = λ1p
    ipeps[xx-1, yy].b = λ2p
    ipeps[xx-1, yy+1].t = λ2p

    return err1, err2, nrm1, nrm2
end


function bond_proj_ru2ld_upPath!(ipeps::iPEPSΓΛ, xx::Int, yy::Int, Dk::Int, gateNNN::TensorMap)
    # 1 site update 方法
    # 先计入交换门
    swgt1 = swap_gate(space(ipeps[xx-1, yy+1].Γ)[3], space(ipeps[xx-1, yy+1].Γ)[1]; Eltype=eltype(ipeps[xx-1, yy+1].Γ))
    swgt2 = swap_gate(space(swgt1)[1], space(ipeps[xx-1, yy+1].Γ)[2]; Eltype=eltype(ipeps[xx-1, yy+1].Γ))
    @tensor Γ1[l1, t1, pu; r1, b1] := ipeps[xx, yy].Γ[l1, te1, pu, re1, be1] * ipeps[xx, yy].b[be1, b1] * ipeps[xx, yy].t[t1, te1] * ipeps[xx, yy].r[re1, r1]
    @tensor Γ3[l3, t3, pd; r3, b3] :=
        ipeps[xx-1, yy+1].Γ[le3in, t3in, pdp, re3, be3] * swgt1[pdpp, le3, pdp, le3in] * swgt2[pd, t3, pdpp, t3in] *
        ipeps[xx-1, yy+1].l[l3, le3] * ipeps[xx-1, yy+1].r[re3, r3] * ipeps[xx-1, yy+1].b[be3, b3]
    # 两次QR
    v1, X1 = rightorth(Γ1, ((1, 3), (2, 4, 5)))
    v3, X3 = rightorth(Γ3, ((2, 3), (1, 4, 5)))
    # 两个bond tensor 与 auxsite 收缩
    @tensor Θ[pd, toΓ3, l2, t2, pmid; pu, toΓ1] :=
        v1[le1, puin, toΓ1] * ipeps[xx, yy].l[l1, le1] * ipeps[xx-1, yy].Γ[le2, te2, pmid, l1, be2] *
        ipeps[xx-1, yy].l[l2, le2] * ipeps[xx-1, yy].t[t2, te2] * ipeps[xx-1, yy].b[be2, b2] * v3[b2, pdin, toΓ3] * gateNNN[pd, pu, pdin, puin]
    # 分出 [xx, yy] 点的 v1，并做截断和归一.
    Θp, λ1p, v1new, err1 = tsvd(Θ, ((1, 2, 3, 4, 5), (6, 7)); trunc=truncdim(Dk), alg=TensorKit.SVD())
    nrm1 = norm(λ1p)
    λ1p = λ1p / nrm1
    # 分出 [xx-1, yy] 点，并做截断和归一
    @tensor Θpp[l2, t2, pmid, r2; pd, toΓ3] := Θp[pd, toΓ3, l2, t2, pmid, r2in] * λ1p[r2in, r2]
    Γ2p, λ2p, v3new, err2 = tsvd(Θpp, ((1, 2, 3, 4), (5, 6)); trunc=truncdim(Dk), alg=TensorKit.SVD())
    nrm2 = norm(λ2p)
    λ2p = λ2p / nrm2
    # 恢复原来的张量及其指标顺序.
    swgtld1 = swap_gate(space(v3new)[2], space(X3)[3]; Eltype=eltype(v3new))
    swgtld2 = swap_gate(space(swgtld1)[1], space(X3)[4]; Eltype=eltype(v3new))
    @tensor Γ1new[l1, t1, pu; r1, b1] := v1new[l1, pu, toΓ1] * X1[toΓ1, t1in, r1in, b1in] * inv(ipeps[xx, yy].b)[b1in, b1] * inv(ipeps[xx, yy].t)[t1, t1in] * inv(ipeps[xx, yy].r)[r1in, r1]
    @tensor Γ2new[l2, t2, pmid; r2, b2] := Γ2p[l2in, t2in, pmid, r2in, b2] * inv(ipeps[xx-1, yy].l)[l2, l2in] * inv(ipeps[xx-1, yy].t)[t2, t2in] * inv(λ1p)[r2in, r2]
    @tensor Γ3new[l3, t3, pd; r3, b3] :=
        v3new[t3, pdin, toΓ3] * swgtld1[pdinp, r3in, pdin, r3inp] * X3[toΓ3, l3in, r3inp, b3inp] * swgtld2[pd, b3in, pdinp, b3inp] *
        inv(ipeps[xx-1, yy+1].l)[l3, l3in] * inv(ipeps[xx-1, yy+1].r)[r3in, r3] * inv(ipeps[xx-1, yy+1].b)[b3in, b3]
    # 更新 tensor
    ipeps[xx, yy].Γ = Γ1new
    ipeps[xx-1, yy].Γ = Γ2new
    ipeps[xx-1, yy+1].Γ = Γ3new
    ipeps[xx, yy].l = λ1p
    ipeps[xx-1, yy].r = λ1p
    ipeps[xx-1, yy].b = λ2p
    ipeps[xx-1, yy+1].t = λ2p

    return err1, err2, nrm1, nrm2
end


"""
更新 [xx, yy] 与 [xx-1, yy+1] 之间的 bond. (次近邻相互作用) See: PRB 82,245119(2010) \n
第二种路径，经过下方 site [xx, yy+1].  \n
Debug hint: [xx, yy] -> 1, [xx, yy+1] -> 2, [xx-1, yy+1] -> 3
"""
function site_proj_ru2ld_dnPath!(ipeps::iPEPSΓΛ, xx::Int, yy::Int, Dk::Int, gateNNN::TensorMap)
    # gate和三个点上的张量都捏起来
    @tensor contractcheck = true Θ[l1, t1, r1, pu; t3, l3, pd, b3, pmid, b2, r2] :=
        gateNNN[pu, pd, puin, pdin] *
        (ipeps[xx, yy].Γ[le1, te1, puin, re1, be1] * ipeps[xx, yy].l[l1, le1] * ipeps[xx, yy].t[t1, te1] * ipeps[xx, yy].r[re1, r1] * ipeps[xx, yy].b[be1, b1]) *
        (ipeps[xx, yy+1].Γ[le2, b1, pmid, re2, be2] * ipeps[xx, yy+1].l[l2, le2] * ipeps[xx, yy+1].r[re2, r2] * ipeps[xx, yy+1].b[be2, b2]) *
        (ipeps[xx-1, yy+1].Γ[le3, te3, pdin, l2, be3] * ipeps[xx-1, yy+1].t[t3, te3] * ipeps[xx-1, yy+1].l[l3, le3] * ipeps[xx-1, yy+1].b[be3, b3])
    # 分出 [xx, yy] 点，并做截断和归一
    Γ1p, λ1p, Θp, err1 = tsvd(Θ, ((1, 2, 3, 4), (5, 6, 7, 8, 9, 10, 11)); trunc=truncdim(Dk))
    nrm1 = norm(λ1p)
    λ1p = λ1p / nrm1
    # 分出 [xx, yy+1] 点，并做截断和归一
    @tensor Θpp[l3, t3, pd, b3; t2, pmid, r2, b2] := λ1p[t2, t2in] * Θp[t2in, t3, l3, pd, b3, pmid, b2, r2]
    Γ3p, λ2p, Γ2p, err2 = tsvd(Θpp, ((1, 2, 3, 4), (5, 6, 7, 8)); trunc=truncdim(Dk))
    nrm2 = norm(λ2p)
    λ2p = λ2p / nrm2
    # 恢复原来的张量及其指标顺序
    # revΓ2l = isomorphism(dual(space(Γ2p)[5]), space(Γ2p)[5])
    @tensor Γ1new[l1, t1, pu; r1, b1] := Γ1p[l1in, t1in, r1in, pu, b1] * inv(ipeps[xx, yy].l)[l1, l1in] * inv(ipeps[xx, yy].t)[t1, t1in] * inv(ipeps[xx, yy].r)[r1in, r1]
    @tensor Γ2new[l2, t2, pmid; r2, b2] := Γ2p[l2, t2, pmid, r2in, b2in] * inv(ipeps[xx, yy+1].r)[r2in, r2] * inv(ipeps[xx, yy+1].b)[b2in, b2]
    @tensor Γ3new[l3, t3, pd; r3, b3] := Γ3p[l3in, t3in, pd, b3in, r3] * inv(ipeps[xx-1, yy+1].t)[t3, t3in] * inv(ipeps[xx-1, yy+1].l)[l3, l3in] * inv(ipeps[xx-1, yy+1].b)[b3in, b3]
    # 更新 tensor
    ipeps[xx, yy].Γ = Γ1new
    ipeps[xx, yy+1].Γ = Γ2new
    ipeps[xx-1, yy+1].Γ = Γ3new
    ipeps[xx, yy].b = λ1p
    ipeps[xx, yy+1].t = λ1p
    ipeps[xx, yy+1].l = λ2p
    ipeps[xx-1, yy+1].r = λ2p

    return err1, err2, nrm1, nrm2
end

function bond_proj_ru2ld_dnPath!(ipeps::iPEPSΓΛ, xx::Int, yy::Int, Dk::Int, gateNNN::TensorMap)
    # 1 site update 方法
    # 先计入交换门
    swgt1 = swap_gate(space(ipeps[xx-1, yy+1].Γ)[3], space(ipeps[xx-1, yy+1].Γ)[5]; Eltype=eltype(ipeps[xx-1, yy+1].Γ))
    swgt2 = swap_gate(space(swgt1)[1], space(ipeps[xx-1, yy+1].Γ)[4]; Eltype=eltype(ipeps[xx-1, yy+1].Γ))
    @tensor Γ1[l1, t1, pu; r1, b1] := ipeps[xx, yy].Γ[le1, te1, pu, re1, b1] * ipeps[xx, yy].l[l1, le1] * ipeps[xx, yy].t[t1, te1] * ipeps[xx, yy].r[re1, r1]
    @tensor Γ3[l3, t3, pd; r3, b3] :=
        ipeps[xx-1, yy+1].Γ[le3, te3, pdin, r3in, be3in] * swgt1[pdinp, be3, pdin, be3in] * swgt2[pd, r3, pdinp, r3in] *
        ipeps[xx-1, yy+1].l[l3, le3] * ipeps[xx-1, yy+1].t[t3, te3] * ipeps[xx-1, yy+1].b[be3, b3]
    # 两次QR
    X1, v1 = leftorth(Γ1, ((1, 2, 4), (3, 5)))
    X3, v3 = leftorth(Γ3, ((1, 2, 5), (3, 4)))
    # 两个bond tensor 与 auxsite 收缩
    swgtmid = swap_gate(space(gateNNN)[1], space(v3)[3]; Eltype=eltype(v3))
    @tensor Θ[toΓ1, pu; toΓ3, pd, pmid, r2, b2] :=
        v1[toΓ1, puin, b1in] * ipeps[xx, yy].b[b1in, b1] * ipeps[xx, yy+1].Γ[le2, b1, pmid, re2, be2] * ipeps[xx, yy+1].l[l2, le2] *
        swgtmid[pd, l2, pdinp, l2in] * ipeps[xx, yy+1].r[re2, r2] * ipeps[xx, yy+1].b[be2, b2] * v3[toΓ3, pdin, l2in] * gateNNN[pdinp, pu, pdin, puin]
    # 分出 [xx, yy] 点的 v1，并做截断和归一.
    v1new, λ1p, Θp, err1 = tsvd(Θ, ((1, 2), (3, 4, 5, 6, 7)); trunc=truncdim(Dk), alg=TensorKit.SVD())
    nrm1 = norm(λ1p)
    λ1p = λ1p / nrm1
    # 分出 [xx, yy+1] 点，并做截断和归一
    @tensor Θpp[toΓ3, pd; t2, pmid, r2, b2] := λ1p[t2, t2in] * Θp[t2in, toΓ3, pd, pmid, r2, b2]
    v3new, λ2p, Γ2p, err2 = tsvd(Θpp, ((1, 2), (3, 4, 5, 6)); trunc=truncdim(Dk), alg=TensorKit.SVD())
    nrm2 = norm(λ2p)
    λ2p = λ2p / nrm2
    # 恢复原来的张量及其指标顺序.
    swgtld = swap_gate(space(v3new)[2], space(X3)[3]; Eltype=eltype(X3))
    @tensor Γ1new[l1, t1, pu; r1, b1] := X1[l1in, t1in, r1in, toΓ1] * v1new[toΓ1, pu, b1] * inv(ipeps[xx, yy].l)[l1, l1in] * inv(ipeps[xx, yy].t)[t1, t1in] * inv(ipeps[xx, yy].r)[r1in, r1]
    @tensor Γ2new[l2, t2, pmid; r2, b2] := Γ2p[l2, t2in, pmid, r2in, b2in] * inv(ipeps[xx, yy+1].r)[r2in, r2] * inv(ipeps[xx, yy+1].b)[b2in, b2] * inv(λ1p)[t2, t2in]
    @tensor Γ3new[l3, t3, pd; r3, b3] :=
        X3[l3in, t3in, b3inp, toΓ3] * v3new[toΓ3, pdin, r3] * swgtld[pd, b3in, pdin, b3inp] *
        inv(ipeps[xx-1, yy+1].l)[l3, l3in] * inv(ipeps[xx-1, yy+1].t)[t3, t3in] * inv(ipeps[xx-1, yy+1].b)[b3in, b3]
    # 更新 tensor
    ipeps[xx, yy].Γ = Γ1new
    ipeps[xx, yy+1].Γ = Γ2new
    ipeps[xx-1, yy+1].Γ = Γ3new
    ipeps[xx, yy].b = λ1p
    ipeps[xx, yy+1].t = λ1p
    ipeps[xx, yy+1].l = λ2p
    ipeps[xx-1, yy+1].r = λ2p

    return err1, err2, nrm1, nrm2
end