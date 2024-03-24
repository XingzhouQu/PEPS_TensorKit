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

    # random Initialization
    Ms = Vector{TensorMap}(undef, Lx * Ly)
    for ii in 1:Lx*Ly
        Ms[ii] = TensorMap(randn, ComplexF64, aspacel ⊗ aspacet ⊗ pspace, aspacer ⊗ aspaceb)
        Ms[ii] = Ms[ii] / norm(Ms[ii])
    end
    @assert Ms[1] != Ms[2]
    ipepsRI = iPEPS(Ms, Lx, Ly)

    # Initialization from ΓΛ 
    ipepsγλ = iPEPSΓΛ(pspace, aspacel, aspacet, Lx, Ly)
    ipepsCI = iPEPS(ipepsγλ)

    @show space(ipepsRI[1, 1])
    @show space(ipepsCI[1, 1])

    for x in 1:Lx, y in 1:Ly
        @assert space(ipepsCI[x, y]) == space(ipepsRI[x, y])
        @assert blocks(ipepsCI[x, y]).keys == blocks(ipepsRI[x, y]).keys
    end

    @show blocks(ipepsCI[1, 1]).keys
    @show blocks(ipepsRI[1, 1]).keys

    envs = iPEPSenv(ipepsCI)
    χ = 20
    Nit = 2
    CTMRG!(ipepsCI, envs, χ, Nit)
    nothing
end

main()