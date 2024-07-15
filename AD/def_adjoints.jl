using Zygote: @adjoint

"""
Custom Gradient: Use Zygote.@adjoint to specify how to compute the gradient of f(x) = x^3 manually.

Zygote.@adjoint f(x) = f(x), Δ -> (3x^2 * Δ,)

The first part, f(x), specifies the value of the function.

The second part, Δ -> (3x^2 * Δ,), specifies how to compute the gradient. 
Δ represents the upstream gradient (the gradient of the loss with respect to the output of f).
"""

# 这里要确保梯度流 Δ 都是 TensorMap 类型？
@adjoint iPEPS(Ms, Lx, Ly) = iPEPS(Ms, Lx, Ly), Δ -> begin
    (iPEPS(Δ .* Ms, Lx, Ly),)
end


@adjoint iPEPSenv(Envs, Lx, Ly) = iPEPSenv(Envs, Lx, Ly), Δ -> begin
    Envs_New = Matrix{_iPEPSenv}(undef, Lx, Ly)
    for xx in 1:Lx, yy in 1:Ly
        innerEnv = Envs[xx, yy]  # type::_iPEPSenv
        corners = innerEnv.corner  # type::_corner
        transfers = innerEnv.transfer  # type::_transfer
        cornersNew = _corner(Δ * corners.lt, Δ * corners.lb, Δ * corners.rt, Δ * corners.rb)
        transfersNew = _transfer(Δ * transfers.l, Δ * transfers.r, Δ * transfers.t, Δ * transfers.b)
        Envs_New[xx, yy] = _iPEPSenv(cornersNew, transfersNew)
    end
    (iPEPSenv(Envs_New, Lx, Ly),)
end
