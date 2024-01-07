using TensorKit
import TensorKit.×

include("../iPEPS/iPEPS.jl")
include("../iPEPS/swap_gate.jl")
include("../iPEPS/util.jl")
include("../CTMRG/CTMRG.jl")

function main()
    pspace = Rep[ℤ₂×SU₂]((0, 0) => 2, (1, 1 // 2) => 1)
    # pspace = Rep[ℤ₂]((1 => 1), (0 => 2))
    aspacel = Rep[ℤ₂×SU₂]((0, 0) => 2, (1, 1 // 2) => 2)
    # aspacer = Rep[ℤ₂×SU₂]((0, 0) => 2, (0, 1) => 2)
    aspacer = aspacel
    aspacet = Rep[ℤ₂×SU₂]((1, 1 // 2) => 2, (0, 1) => 2)
    # aspaceb = Rep[ℤ₂×SU₂]((0, 0) => 2, (1, 3 // 2) => 2)
    aspaceb = aspacet
    A = TensorMap(randn, ComplexF64, aspacel ⊗ aspacet ⊗ pspace, aspacer ⊗ aspaceb)
    Ms = [A, A, A, A]
    ipeps = iPEPS(Ms, 2, 2)

    envs = iPEPSenv(ipeps)
    χ = 10
    Nit = 2
    CTMRG!(ipeps, envs, χ, Nit)
end

main()