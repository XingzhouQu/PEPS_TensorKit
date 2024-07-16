module SU2spin

using TensorKit

const pspace = Rep[SU₂](1 // 2 => 1)

const SS = let
    aspace = Rep[SU₂](1 => 1)
    SL = TensorMap(ones, Float64, pspace, pspace ⊗ aspace) * sqrt(3) / 2
    SR = permute(SL', ((2, 1), (3,)))

    @tensor SS[p1, p3; p2, p4] := SL[p1, p2, a] * SR[a, p3, p4]
    SS
end

end

const SU₂spin = SU2spin

function Heisenberg_hij(para::Dict{Symbol,Any})
    J = para[:J]
    ss = SU2spin.SS
    gate = J * ss
    return [gate]
end

function get_op_Heisenberg(tag::String, para::Dict{Symbol,Any})
    if tag == "hij"
        return Heisenberg_hij(para)
    else
        error("Unsupported tag. Check input tag or add this operator.")
    end
end