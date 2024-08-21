#  =================== TensorKit.jl

# you cannot really define the pullback wrt a function, because you don't have an inner product
@non_differentiable TensorKit.TensorMap(f::Function, storagetype, cod, dom)

@non_differentiable TensorKit.isomorphism(args...)
@non_differentiable TensorKit.isometry(args...)

function ChainRulesCore.rrule(::typeof(tr), A::AbstractTensorMap)
    function tr_pushback(f̄wd)
        ∂A = @thunk(f̄wd * id(domain(A)))
        return NoTangent(), ∂A
    end
    return tr(A), tr_pushback
end

function ChainRulesCore.rrule(::typeof(adjoint), A::AbstractTensorMap)
    function adjoint_pushback(f̄wd)
        return NoTangent(), f̄wd'
    end
    return A', adjoint_pushback
end

function ChainRulesCore.rrule(::typeof(dot), a::AbstractTensorMap, b::AbstractTensorMap)
    function pullback(c)
        ∂a = @thunk(b * c')
        ∂b = @thunk(a * c)
        return (NoTangent(), ∂a, ∂b)
    end
    return dot(a, b), pullback
end

function ChainRulesCore.rrule(::typeof(+), a::AbstractTensorMap, b::AbstractTensorMap)
    pullback(c) = (NoTangent(), c, c)
    return a + b, pullback
end

function ChainRulesCore.rrule(::typeof(-), a::AbstractTensorMap, b::AbstractTensorMap)
    pullback(c) = (NoTangent(), c, -c)
    return a - b, pullback
end

function ChainRulesCore.rrule(::typeof(norm), a::AbstractTensorMap, p)
    p == 2 || throw(ArgumentError("DIY"))

    na = norm(a)
    function pullback(c)
        ∂a = @thunk(a * (c' + c) / (na * 2))
        return (NoTangent(), ∂a)
    end
    return na, pullback
end

function ChainRulesCore.rrule(::typeof(*), a::AbstractTensorMap, b::AbstractTensorMap)
    function pullback(c)
        ∂a = @thunk(c * b')
        ∂b = @thunk(a' * c)
        return (NoTangent(), ∂a, ∂b)
    end
    return a * b, pullback
end

function ChainRulesCore.rrule(::typeof(*), a::AbstractTensorMap, b::Number)
    function pullback(c)
        ∂a = @thunk(c * b')
        ∂b = @thunk(dot(a, c))
        return (NoTangent(), ∂a, ∂b)
    end
    return a * b, pullback
end

function ChainRulesCore.rrule(::typeof(*), a::Number, b::AbstractTensorMap)
    function pullback(c)
        ∂a = @thunk(dot(b, c))
        ∂b = @thunk(a' * c)
        return (NoTangent(), ∂a, ∂b)
    end
    return a * b, pullback
end

function ChainRulesCore.rrule(::typeof(permute), tensor::AbstractTensorMap, leftind, rightind=())
    function pullback(c)
        ∂a = @thunk begin
            invperm = TupleTools.invperm(tuple(leftind..., rightind...))

            permute(c, tuple(invperm[1:numout(tensor)]...), tuple(invperm[numout(tensor)+1:end]...))
        end
        return (NoTangent(), ∂a, NoTangent(), NoTangent())
    end

    return permute(tensor, leftind, rightind), pullback
end

function ChainRulesCore.rrule(::Type{<:TensorMap}, d::DenseArray, args...)
    function pullback(tm)
        ∂d = @thunk(convert(Array, tm))
        (NoTangent(), ∂d, [NoTangent() for a in args]...)
    end
    TensorMap(d, args...), pullback
end

function ChainRulesCore.rrule(::typeof(convert), ::Type{<:Array}, t::AbstractTensorMap)
    function pullback(tm)
        spacetype(t) <: ComplexSpace || throw(ArgumentError("not yet implemented"))
        ∂d = TensorMap(tm, codomain(t), domain(t))
        (NoTangent(), NoTangent(), ∂d)
    end

    convert(Array, t), pullback
end

# pullback rule based on tom's krylovkit rule
function ChainRulesCore.rrule(::typeof(TensorKit.tsvd), t::AbstractTensorMap; kwargs...)
    T = eltype(t)

    (U, S, V) = tsvd(t; kwargs...)

    F = similar(S)
    for (k, dst) in blocks(F)

        src = blocks(S)[k]

        @inbounds for i in 1:size(dst, 1), j in 1:size(dst, 2)
            if abs(src[j, j] - src[i, i]) < 1e-12
                d = 1e-12
            else
                d = src[j, j]^2 - src[i, i]^2
            end

            dst[i, j] = (i == j) ? zero(eltype(S)) : 1 / d
        end
    end


    function pullback(v)
        dU, dS, dV = v

        dA = zero(t)
        #A_s bar term
        if dS != ChainRulesCore.ZeroTangent()
            dA += U * _elementwise_mult(dS, one(dS)) * V
        end
        #A_uo bar term
        if dU != ChainRulesCore.ZeroTangent()
            J = _elementwise_mult((U' * dU), F)
            dA += U * (J + J') * S * V
        end
        #A_vo bar term
        if dV != ChainRulesCore.ZeroTangent()
            VpdV = V * dV'
            K = _elementwise_mult(VpdV, F)
            dA += U * S * (K + K') * V
        end
        #A_d bar term, only relevant if matrix is complex
        if dV != ChainRulesCore.ZeroTangent() && T <: Complex
            L = _elementwise_mult(VpdV, one(F))
            dA += 1 / 2 * U * pinv(S) * (L' - L) * V
        end

        if codomain(t) != domain(t)
            pru = U * U'
            prv = V' * V
            dA += (one(pru) - pru) * dU * pinv(S) * V
            dA += U * pinv(S) * dV * (one(prv) - prv)
        end

        return NoTangent(), dA, [NoTangent() for kwa in kwargs]...
    end
    return (U, S, V), pullback
end


function _elementwise_mult(a::AbstractTensorMap, b::AbstractTensorMap)
    dst = similar(a)
    for (k, block) in blocks(dst)
        copyto!(block, blocks(a)[k] .* blocks(b)[k])
    end
    dst
end


function ChainRulesCore.rrule(::typeof(leftorth), tensor, leftind=codomainind(tensor), rightind=domainind(tensor); alg=QRpos())

    (permuted, permback) = ChainRulesCore.rrule(permute, tensor, leftind, rightind)
    (q, r) = leftorth(permuted; alg=alg)

    if alg isa TensorKit.QR || alg isa TensorKit.QRpos
        pullback = v -> backwards_leftorth_qr(permuted, q, r, v[1], v[2])
    else
        pullback = v -> @assert false
    end

    (q, r), v -> (NoTangent(), permback(pullback(v))[2], NoTangent(), NoTangent(), NoTangent())
end

function backwards_leftorth_qr(A, q, r, dq, dr)
    out = similar(A)
    dr = dr isa ZeroTangent ? zero(r) : dr
    dq = dq isa ZeroTangent ? zero(q) : dq


    if sectortype(A) == Trivial
        copyto!(out.data, qr_back(A.data, q.data, r.data, dq.data, dr.data))
    else
        for b in keys(blocks(A))
            cA = A[b]
            cq = q[b]
            cr = r[b]
            cdq = dq[b]
            cdr = dr[b]

            copyto!(out[b], qr_back(cA, cq, cr, cdq, cdr))
        end
    end
    out
end

function ChainRulesCore.rrule(::typeof(rightorth), tensor, leftind=codomainind(tensor), rightind=domainind(tensor); alg=LQpos())

    (permuted, permback) = ChainRulesCore.rrule(permute, tensor, leftind, rightind)
    (l, q) = rightorth(permuted; alg=alg)

    if alg isa TensorKit.LQ || alg isa TensorKit.LQpos
        pullback = v -> backwards_rightorth_lq(permuted, l, q, v[1], v[2])
    else
        pullback = v -> @assert false
    end

    (l, q), v -> (NoTangent(), permback(pullback(v))[2], NoTangent(), NoTangent(), NoTangent())
end

function backwards_rightorth_lq(A, l, q, dl, dq)
    out = similar(A)
    dl = dl isa ZeroTangent ? zero(l) : dl
    dq = dq isa ZeroTangent ? zero(q) : dq

    if sectortype(A) == Trivial
        copyto!(out.data, lq_back(A.data, l.data, q.data, dl.data, dq.data))
    else
        for b in keys(blocks(A))
            cA = A[b]
            cl = l[b]
            cq = q[b]
            cdl = dl[b]
            cdq = dq[b]

            copyto!(out[b], lq_back(cA, cl, cq, cdl, cdq))
        end
    end
    out
end


function ChainRulesCore.rrule(::typeof(Base.convert), ::Type{Dict}, t::AbstractTensorMap)
    out = convert(Dict, t)
    function pullback(c)

        if haskey(c, :data) # :data is the only thing for which this dual makes sense
            dual = copy(out)
            dual[:data] = c[:data]
            return (NoTangent(), NoTangent(), convert(TensorMap, dual))
        else
            # instead of zero(t) you can also return ZeroTangent(), which is type unstable
            return (NoTangent(), NoTangent(), zero(t))
        end

    end
    out, pullback
end
ChainRulesCore.rrule(::typeof(Base.convert), ::Type{TensorMap}, t::Dict{Symbol,Any}) = convert(TensorMap, t), v -> (NoTangent(), NoTangent(), convert(Dict, v))


# ======================== KrylovKit.jl

function ChainRulesCore.rrule(
    ::typeof(KrylovKit.linsolve),
    A::AbstractMatrix,
    b::AbstractVector,
    x₀,
    algorithm,
    a₀,
    a₁
)
    (x, info) = KrylovKit.linsolve(A, b, x₀, algorithm, a₀, a₁)


    function linsolve_pullback(x̄)
        ∂self = NoTangent()
        ∂x₀ = ZeroTangent()
        ∂algorithm = NoTangent()
        (∂b, _) = KrylovKit.linsolve(
            A', x̄[1], (zero(a₀) * zero(a₁)) * x̄[1], algorithm, a₀, a₁
        )
        ∂a₀ = -dot(x, ∂b)
        ∂A = -a₁ * ∂b * x'
        ∂a₁ = -x' * A' * ∂b
        return ∂self, ∂A, ∂b, ∂x₀, ∂algorithm, ∂a₀, ∂a₁
    end
    return (x, info), linsolve_pullback
end



function ChainRulesCore.rrule(
    config::RuleConfig{>:HasReverseMode},
    ::typeof(KrylovKit.linsolve),
    f,
    b,
    x₀,
    algorithm,
    a₀,
    a₁
)
    x, info = KrylovKit.linsolve(f, b, x₀, algorithm, a₀, a₁)

    # f defines a linear map => pullback defines action of the adjoint
    # TODO this is probably not necessary if self-adjoint, see kwargs.
    (y, f_pullback) = rrule_via_ad(config, f, x)
    fᵀ(xᵀ) = f_pullback(xᵀ)[2]

    function linsolve_pullback(x̄)
        ∂self = NoTangent()
        ∂x₀ = ZeroTangent()
        ∂algorithm = NoTangent()
        (∂b, _) = KrylovKit.linsolve(
            fᵀ, x̄[1], (zero(a₀) * zero(a₁)) * x̄[1], algorithm, a₀, a₁
        )
        ∂a₀ = -x' * ∂b
        ∂f = a₁ * ∂a₀
        ∂a₁ = -y' * ∂b
        return ∂self, ∂f, ∂b, ∂x₀, ∂algorithm, ∂a₀, ∂a₁
    end
    return (x, info), linsolve_pullback
end

function ChainRulesCore.rrule(::typeof(KrylovKit.eigsolve), A::AbstractMatrix, x₀, howmany, which, algorithm)
    @show vals, vecs, info = eigsolve(A, x₀, howmany, which, algorithm)

    function eigsolve_pullback(Q̄)
        v̄als, v̄ecs, _ = Q̄
        ∂self = NoTangent()
        ∂x₀ = ZeroTangent()
        ∂which = NoTangent()
        ∂algorithm = NoTangent()
        ∂howmany = NoTangent()

        ∂A = map(1:howmany) do i
            α = vals[i]
            x = vecs[i]
            ᾱ = v̄als[i]
            x̄ = v̄ecs[i]
            @show λ₀ = _eigsolve_λ₀(A, α, x, x̄)
            return -λ₀ * x' + ᾱ * x * x'
        end
        return ∂self, collect(∂A), ∂x₀, ∂howmany, ∂which, ∂algorithm
    end
    return (vals, vecs, info), eigsolve_pullback
end
using LinearAlgebra
function LinearAlgebra.norm(x::ZeroTangent)
    return 0
end

function _eigsolve_λ₀(A::AbstractMatrix, α, x, x̄::ZeroTangent)
    return x̄
end

function _eigsolve_λ₀(A::AbstractMatrix, α, x, x̄)
    RL = (I - x * x') * x̄
    #RL = RL - dot(x, RL) * RL

    f(y) = A * y - α * y
    λ₀, _ = linsolve(f, RL)
    @show dot(x, λ₀)
    return λ₀ - dot(x, λ₀) * λ₀
end

# ========================= backwardslinalg.jl
struct ZeroAdder end
Base.:+(a, zero::ZeroAdder) = a
Base.:+(zero::ZeroAdder, a) = a
Base.:-(a, zero::ZeroAdder) = a
Base.:-(zero::ZeroAdder, a) = -a
Base.:-(zero::ZeroAdder) = zero

"""
		qr(A) -> Tuple{AbstractMatrix, AbstractMatrix}
private QR method, call LinearAlgebra.qr to achieve forward calculation, while
return Tuple type
"""
function qr(A)
    res = LinearAlgebra.qr(A)
    Matrix(res.Q), res.R
end

"""
		qr(A, pivot) -> Tuple{AbstractMatrix, AbstractMatrix, AbstractVector}
"""
function qr(A::AbstractMatrix, pivot::Val{true})
    res = LinearAlgebra.qr(A, pivot)
    Q, R, P = Matrix(res.Q), res.R, res.P
end

"""
		lq(A) -> Tuple{AbstractMatrix, AbstractMatrix}
"""
function lq(A)
    res = LinearAlgebra.lq(A)
    res.L, Matrix(res.Q)
end


function trtrs!(c1::Char, c2::Char, c3::Char, r::AbstractMatrix, b::AbstractVecOrMat)
    LinearAlgebra.LAPACK.trtrs!(c1, c2, c3, r, b)
end

"""
    copyltu!(A::AbstractMatrix) -> AbstractMatrix
copy the lower triangular to upper triangular.
"""
function copyltu!(A::AbstractMatrix)
    m, n = size(A)
    for i = 1:m
        A[i, i] = real(A[i, i])
        for j = i+1:n
            @inbounds A[i, j] = conj(A[j, i])
        end
    end
    A
end

"""
    qr_back_fullrank(q, r, dq, dr) -> Matrix
backward for QR decomposition, for input matrix (in forward pass) with M > N.
References:
    Seeger, M., Hetzel, A., Dai, Z., Meissner, E., & Lawrence, N. D. (2018). Auto-Differentiating Linear Algebra.
"""
function qr_back_fullrank(q, r, dq, dr)
    dqnot0 = !(typeof(dq) <: Nothing)
    drnot0 = !(typeof(dr) <: Nothing)
    (!dqnot0 && !drnot0) && return nothing
    ex = drnot0 && dqnot0 ? (r * dr' - dq' * q) : (dqnot0 ? (-dq' * q) : (r * dr'))
    b = (dqnot0 ? dq : ZeroAdder()) + q * copyltu!(ex)
    trtrs!('U', 'N', 'N', r, do_adjoint(b))'
end

do_adjoint(A::Matrix) = Matrix(A')

"""
    qr_back(A, q, r, dq, dr) -> Matrix
backward for QR decomposition, for an arbituary shaped input matrix.
References:
    Seeger, M., Hetzel, A., Dai, Z., Meissner, E., & Lawrence, N. D. (2018). Auto-Differentiating Linear Algebra.
    Differentiable Programming Tensor Networks, Hai-Jun Liao, Jin-Guo Liu, Lei Wang, Tao Xiang
"""
function qr_back(A, oq, or, odq, odr)
    dqnot0 = !(typeof(odq) <: Nothing)
    drnot0 = !(typeof(odr) <: Nothing)
    (!dqnot0 && !drnot0) && return nothing

    N = size(or, 2)
    M = 0
    ref = or[1, 1]
    while M < size(or, 1)
        abs(or[M+1, M+1] / ref) < 1e-12 && break
        M += 1
    end

    q = view(oq, :, 1:M)
    dq = dqnot0 ? view(odq, :, 1:M) : odq
    r = view(or, 1:M, :)
    dr = drnot0 ? view(odr, 1:M, :) : odr

    N == M && return qr_back_fullrank(q, r, dq, dr)

    B = view(A, :, M+1:N)
    U = view(r, :, 1:M)
    D = view(r, :, M+1:N)
    if drnot0
        dD = view(dr, :, M+1:N)
        da = qr_back_fullrank(q, U, dqnot0 ? (dq + B * dD') : (B * dD'), view(dr, :, 1:M))
        db = q * dD
    else
        da = qr_back_fullrank(q, U, dq, nothing)
        db = zero(B)
    end
    hcat(da, db)
end

"""
    lq_back(A, l, q, dl, dq) -> Matrix
backward for LQ decomposition, for an arbituary shaped input matrix.
References:
    Seeger, M., Hetzel, A., Dai, Z., Meissner, E., & Lawrence, N. D. (2018). Auto-Differentiating Linear Algebra.
    Differentiable Programming Tensor Networks, Hai-Jun Liao, Jin-Guo Liu, Lei Wang, Tao Xiang
"""
function lq_back_fullrank(L, Q, dL, dQ)
    M = ZeroAdder()
    dL === nothing || (M += L' * dL)
    dQ === nothing || (M -= dQ * Q')
    C = copyltu!(M) * Q
    if dQ !== nothing
        C += dQ
    end
    #inv(L)' * C
    trtrs!('L', 'C', 'N', L, C)
end

"""
    lq_back(A, L, Q, dL, dQ) -> Matrix
backward for QR decomposition, for an arbituary shaped input matrix.
References:
    Seeger, M., Hetzel, A., Dai, Z., Meissner, E., & Lawrence, N. D. (2018). Auto-Differentiating Linear Algebra.
    HaiJun's paper.
"""
function lq_back(A, oL, oQ, odL, odQ)
    dunot0 = !(typeof(odQ) <: Nothing)
    dlnot0 = !(typeof(odL) <: Nothing)
    (!dunot0 && !dlnot0) && return nothing

    N = size(oL, 1)
    M = 0
    ref = oL[1, 1]
    while M < size(oL, 2)
        abs(oL[M+1, M+1] / ref) < 1e-12 && break
        M += 1
    end

    L = view(oL, :, 1:M)
    dL = dlnot0 ? view(odL, :, 1:M) : odL
    Q = view(oQ, 1:M, :)
    dQ = dunot0 ? view(odQ, 1:M, :) : odQ

    M == N && return lq_back_fullrank(L, Q, dL, dQ)
    B = view(A, M+1:N, :)
    U = view(L, 1:M, :)
    D = view(L, M+1:N, :)
    if dlnot0
        dD = view(dL, M+1:N, :)
        da = lq_back_fullrank(U, Q, view(dL, 1:M, :), dunot0 ? (dQ + dD' * B) : (dD' * B))
        db = dD * Q
    else
        da = lq_back_fullrank(U, Q, nothing, dQ)
        db = zero(B)
    end
    vcat(da, db)
end
