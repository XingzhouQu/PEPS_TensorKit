module U1spin

using TensorKit

const pspace = Rep[U₁](-1 // 2 => 1, 1 // 2 => 1)

const Sz = let
    Sz = TensorMap(ones, pspace, pspace)
    block(Sz, Irrep[U₁](1 // 2)) .= 1 / 2
    block(Sz, Irrep[U₁](-1 // 2)) .= -1 / 2
    Sz
end

# S+ S- interaction
# convention: S⋅S = SzSz + (S₊₋ + S₋₊)/2
const S₊₋ = let
    aspace = Rep[U₁](1 => 1)
    S₊ = TensorMap(ones, pspace, pspace ⊗ aspace)
    S₋ = TensorMap(ones, aspace ⊗ pspace, pspace)

    @tensor S₊S₋[p1, p3; p2, p4] := S₊[p1, p2, a] * S₋[a, p3, p4]
    S₊S₋
end

const S₋₊ = let
    aspace = Rep[U₁](1 => 1)
    iso = isometry(aspace, flip(aspace))
    Sp = TensorMap(ones, pspace, pspace ⊗ aspace)
    Sm = TensorMap(ones, aspace ⊗ pspace, pspace)
    @tensor S₋[a; c d] := Sp'[a, b, c] * iso'[d, b]
    @tensor S₊[d a; c] := Sm'[a, b, c] * iso[b, d]

    @tensor S₋S₊[p1, p3; p2, p4] := S₋[p1, p2, a] * S₊[a, p3, p4]
    S₋S₊
end

end

const U₁spin = U1spin

function Heisenberg_hij(para::Dict{Symbol,Any})
    J = para[:J]
    ss = U1spin.Sz ⊗ U1spin.Sz + (U1spin.S₊₋ + U1spin.S₋₊) / 2
    gate = J * ss
    return [gate]
end

function J1J2_Heisenberg_hij(para::Dict{Symbol,Any})
    J1 = para[:J1]
    J2 = para[:J2]
    ss = U1spin.Sz ⊗ U1spin.Sz + (U1spin.S₊₋ + U1spin.S₋₊) / 2
    gateNN = J1 * ss
    gateNNN = J2 * ss
    return [gateNN, gateNNN]
end

function get_op_Heisenberg(tag::String, para::Dict{Symbol,Any})
    if tag == "hij"
        return Heisenberg_hij(para)[1]
    elseif tag == "hijNN"
        return J1J2_Heisenberg_hij(para)[1]
    elseif tag == "hijNNN"
        return J1J2_Heisenberg_hij(para)[2]
    elseif tag == "Sz"
        return U1spin.Sz
    elseif tag == "SzSz"
        return U1spin.Sz ⊗ U1spin.Sz
    elseif tag == "SpSm"
        return U1spin.S₊₋
    else
        error("Unsupported tag. Check input tag or add this operator.")
    end
end