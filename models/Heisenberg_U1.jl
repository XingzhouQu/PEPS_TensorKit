function Sz(pspace::GradedSpace)
    Sz = TensorMap(ones, pspace, pspace)
    block(Sz, Irrep[U₁](1 // 2)) .= 1 / 2
    block(Sz, Irrep[U₁](-1 // 2)) .= -1 / 2
    return Sz
end

# S+ S- interaction
# convention: S⋅S = SzSz + (S₊₋ + S₋₊)/2
function S₊₋(pspace::GradedSpace)
    aspace = Rep[U₁](1 => 1)
    S₊ = TensorMap(ones, pspace, pspace ⊗ aspace)
    S₋ = TensorMap(ones, aspace ⊗ pspace, pspace)

    @tensor S₊S₋[p1, p3; p2, p4] := S₊[p1, p2, a] * S₋[a, p3, p4]
    return S₊S₋
end

function S₋₊(pspace::GradedSpace)
    aspace = Rep[U₁](1 => 1)
    iso = isometry(aspace, flip(aspace))
    Sp = TensorMap(ones, pspace, pspace ⊗ aspace)
    Sm = TensorMap(ones, aspace ⊗ pspace, pspace)
    @tensor S₋[a; c d] := Sp'[a, b, c] * iso'[d, b]
    @tensor S₊[d a; c] := Sm'[a, b, c] * iso[b, d]

    @tensor S₋S₊[p1, p3; p2, p4] := S₋[p1, p2, a] * S₊[a, p3, p4]
    return S₋S₊
end

function Heisenberg_hij(para::Dict{Symbol,Any})
    J = para[:J]
    pspace = para[:pspace]
    ss = Sz(pspace) ⊗ Sz(pspace) + (S₊₋(pspace) + S₋₊(pspace)) / 2
    gate = J * ss
    return gate
end

function get_op_Heisenberg(tag::String, para::Dict{Symbol,Any})
    if tag == "hij"
        return Heisenberg_hij(para)
    elseif tag == "Sz"
        return Sz(para[:pspace])
    elseif tag == "SzSz"
        return Sz(para[:pspace]) ⊗ Sz(para[:pspace])
    elseif tag == "SpSm"
        return S₊₋(para[:pspace])
    else
        error("Unsupported tag. Check input tag or add this operator.")
    end
end