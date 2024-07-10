using Zygote
using ChainRulesCore
using OptimKit

"""
OptimKit.jl usage:

`x, fx, gx, numfg, normgradhistory = optimize(fg, x₀, algorithm; kwargs...)`

where `fval, gval = fg(x)` is a function that returns both the function value `fval` and the gradient `gval`
at a given point `x`.

We need to prepare the `fg(x)` function.

In terms of CTMRG at fixed point, the target function named by `loss_Energy(ipeps, envs)`
returns nothing but the expectation value of energy ⟨H⟩ and the gradient of iPEPS tensors ∂⟨H⟩/∂A.

"""
function loss_Energy(ipeps::iPEPS, envs::iPEPSEnv, Cal_Obs::Function, CTMRG::Function, para::Dict{Symbol,Any})
    Lx = ipeps.Lx
    Ly = ipeps.Ly
    # 以下符号简记：A → ipeps张量，e → ipeps环境张量，ε → 求能量的函数，c → CMTRG函数
    # ∂⟨H⟩/∂A = ∂ε/∂A + ∂ε/∂e * ∂e/∂A = ∂ε/∂A + ∂ε/∂e * (1 - K)⁻¹ * L
    # K = ∂c/∂e,  L = ∂c/∂A

    return energy, ∂H∂A
end


function ()

end
