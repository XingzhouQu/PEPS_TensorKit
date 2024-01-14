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