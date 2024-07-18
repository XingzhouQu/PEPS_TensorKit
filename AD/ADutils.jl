using Zygote
using ChainRulesCore: ignore_derivatives
using OptimKit

"""
OptimKit.jl usage:

`x, fx, gx, numfg, normgradhistory = optimize(fg, x₀, algorithm; kwargs...)`

where `fval, gval = fg(x)` is a function that returns both the function value `fval` and the gradient `gval`
at a given point `x`.

We need to prepare the `fg(x)` function.

In terms of CTMRG at fixed point, the target function named by `loss_Energy(ipeps, envs)`
returns nothing but the expectation value of energy ⟨H⟩ and the gradient of iPEPS tensors ∂⟨H⟩/∂A.

需要根据能量期望值的计算对内部略加修改
"""
function loss_Energy(ipeps::iPEPS, envs::iPEPSenv, get_op::Function, CTMRG::Function, para::Dict{Symbol,Any})
    # 求梯度. 以下符号简记：A → ipeps张量，e → ipeps环境张量，ε → 求能量的函数，c → CMTRG函数

    E, grad = withgradient(ipeps, envs) do peps, env
        Cal_Energy(peps, env, get_op, para)
    end
    ∂ε_∂A, ∂ε_∂e = grad

    # TODO 下面求K，L都是形式上写的模板，实现这些函数和方法
    # K = ∂c/∂e,  L = ∂c/∂A 都是 Jacobian.
    _, K = pullback() do

    end

    _, L = pullback() do

    end

    # ∂⟨H⟩/∂A = ∂ε/∂A + ∂ε/∂e * ∂e/∂A = ∂ε/∂A + ∂ε/∂e * (1 - K)⁻¹ * L
    ∂H_∂A = ∂ε_∂A + ∂ε_∂e * inv(1 - K) * L
    return E, ∂H_∂A
end


function Cal_Energy(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, get_op::Function, para::Dict{Symbol,Any})
    # 初始化
    Lx = ipeps.Lx::Int
    Ly = ipeps.Ly::Int

    # 算能量
    Etot = 0.0
    for ind in CartesianIndices((Lx, Ly))
        (xx, yy) = Tuple(ind)
        Ebond2 = _2siteObs_adjSite(ipeps, ignore_derivatives(ipepsbar), envs, ["hijNN"], para, [xx, yy], [xx + 1, yy], get_op; ADflag=true)
        Etot += Ebond2
        Ebond = _2siteObs_adjSite(ipeps, ignore_derivatives(ipepsbar), envs, ["hijNN"], para, [xx, yy], [xx, yy + 1], get_op; ADflag=true)
        Etot += Ebond
    end
    E = Etot / (Lx * Ly)  # TODO 这里的能量暂时没有扣除化学势的贡献，这样做对吗？
    return E
end

"""
现存的 CTMRG 函数是不断做 in-place 更新. 形如 `CTMRG!(ipeps, envs)` 迭代更改 iPEPSenv struct 里面的内容.

但是 Zygote.jl 不支持 `f!(x)` 这样的形式，需要套一层用于 AD 的皮，显式指定返回值，改为 `y = f(x)`形式.

可能会带来更多内存分配.

详见 `Zygote.Buffer()` 和 https://fluxml.ai/Zygote.jl/latest/limitations/

此函数 `e = Non_mutating_Wrapper_CTM(A, e)`

Fermionic 版本还要求输入 ipepsbar, 但是不计入梯度计算
"""
# TODO: 实在不行就写一个无 mutating 的 envs = CTMRG(ipeps, envs), 仅用于自动微分。。。
function Non_mutating_Wrapper_CTM(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, para::Dict{Symbol,Any})
    para = ignore_derivatives(para)
    ipepsbar = ignore_derivatives(ipepsbar)
    CTMRG!(ipeps::iPEPS, ipepsbar::iPEPS, envs::iPEPSenv, para[:χ], para[:Nit]; parallel=false)
    return envs
end
function Non_mutating_Wrapper_CTM(ipeps::iPEPS, envs::iPEPSenv, para::Dict{Symbol,Any})
    para = ignore_derivatives(para)
    CTMRG!(ipeps::iPEPS, envs::iPEPSenv, para[:χ], para[:Nit]; parallel=false)
    return envs
end

include("./def_adjoints.jl")