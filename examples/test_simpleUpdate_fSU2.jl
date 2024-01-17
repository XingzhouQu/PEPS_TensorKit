using TensorKit
import TensorKit.×

include("../iPEPS_fSU2/iPEPS.jl")
include("../CTMRG_fSU2/CTMRG.jl")
include("../simple_update_fSU2/simple_update.jl")

function main()
    para = Dict{Symbol,Any}()
    para[:t] = 1.0
    para[:U] = 8
    para[:τlis] = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3]
    para[:Dk] = 10  # Dkept in the simple udate

    pspace = GradedSpace{fSU₂}((0 => 2), (1 // 2 => 1))
    aspacelr = pspace
    aspacetb = pspace
    Lx = 2
    Ly = 2
    ipeps = iPEPSΓΛ(pspace, aspacelr, aspacetb, Lx, Ly; dtype=ComplexF64)
    simple_update!(ipeps, para)


end

main()