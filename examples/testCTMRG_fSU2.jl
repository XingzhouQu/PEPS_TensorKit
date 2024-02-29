using TensorKit
import TensorKit.×

include("../iPEPS_fSU2/iPEPS.jl")
include("../CTMRG_fSU2/CTMRG.jl")

function main()
    pspace = GradedSpace{fSU₂}((0 => 2), (1 // 2 => 1))
    # pspace = Rep[ℤ₂]((1 => 1), (0 => 2))
    aspacel = GradedSpace{fSU₂}((0 => 2), (1 // 2 => 1))
    # aspacer = Rep[ℤ₂×SU₂]((0, 0) => 2, (0, 1) => 2)
    aspacer = aspacel
    aspacet = GradedSpace{fSU₂}((0 => 2), (1 // 2 => 1))
    # aspaceb = Rep[ℤ₂×SU₂]((0, 0) => 2, (1, 3 // 2) => 2)
    aspaceb = aspacet
    A = TensorMap(randn, ComplexF64, aspacel ⊗ aspacet ⊗ pspace, aspacer ⊗ aspaceb)
    Ms = [A, A, A, A]
    ipeps = iPEPS(Ms, 2, 2)

    envs = iPEPSenv(ipeps)
    χ = 20
    Nit = 2
    CTMRG!(ipeps, envs, χ, Nit)
end

main()