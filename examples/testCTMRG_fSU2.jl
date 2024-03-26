using TensorKit
import TensorKit.×

include("../iPEPS_fSU2/iPEPS.jl")
include("../CTMRG_fSU2/CTMRG.jl")

function main()
    Lx = 2
    Ly = 2
    pspace = GradedSpace{fSU₂}((0 => 2), (1 // 2 => 1))
    # pspace = Rep[ℤ₂]((1 => 1), (0 => 2))
    aspacel = GradedSpace{fSU₂}((0 => 2), (1 // 2 => 1))
    # aspacer = Rep[ℤ₂×SU₂]((0, 0) => 2, (0, 1) => 2)
    aspacer = aspacel
    aspacet = GradedSpace{fSU₂}((0 => 2), (1 // 2 => 1))
    # aspaceb = Rep[ℤ₂×SU₂]((0, 0) => 2, (1, 3 // 2) => 2)
    aspaceb = aspacet

    aspacelr = pspace
    aspacetb = pspace

    # random Initialization
    Ms = Vector{TensorMap}(undef, Lx * Ly)
    for ii in 1:Lx*Ly
        Ms[ii] = TensorMap(randn, ComplexF64, aspacelr ⊗ aspacetb ⊗ pspace, aspacelr ⊗ aspacetb)
        Ms[ii] = Ms[ii] / norm(Ms[ii])
    end
    # @assert Ms[1] != Ms[2]
    ipepsRI = iPEPS(Ms, Lx, Ly)

    # Initialization from ΓΛ 
    ipepsγλ = iPEPSΓΛ(pspace, aspacel, aspacet, Lx, Ly)
    ipepsCI = iPEPS(ipepsγλ)

    envs = iPEPSenv(ipepsRI)

    check_qn(ipepsCI, envs)
    @assert envs[0, 0].transfer.l == envs[Lx, Ly].transfer.l

    χ = 20
    Nit = 2
    CTMRG!(ipepsCI, envs, χ, Nit)
    nothing
end

main()