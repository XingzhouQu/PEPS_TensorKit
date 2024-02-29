using TensorKit
import TensorKit.×

function main()
    pspacef = GradedSpace{fℤ₂}((0 => 1), (1 => 1))
    pspace = Rep[ℤ₂]((0 => 1), (1 => 1))
    Af = TensorMap(ones, Float64, pspacef ⊗ pspacef, pspacef)
    A = TensorMap(ones, Float64, pspace ⊗ pspace, pspace)
    @tensor ContractF[:] := Af[a, b, c] * Af'[c, b, a]
    @planar @tensor ContractFp[:] := Af[a, b, c] * Af'[c, b, a]
    @tensor Contract[:] := A[a, b, c] * A'[c, b, a]
    println("Use fℤ₂ gives $ContractF")
    println("Use fℤ₂ + planar gives $ContractFp")
    println("Use ℤ₂ gives $Contract")
end

main()