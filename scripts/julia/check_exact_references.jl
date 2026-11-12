# Optional, standard-library-only cross-check of three explicit quantum witnesses.
# Run: julia --startup-file=no --history-file=no scripts/julia/check_exact_references.jl
using LinearAlgebra

const Q = Rational{BigInt}
const CQ = Complex{Q}
q(n, d=1) = BigInt(n) // BigInt(d)

const I2 = Matrix{CQ}(I, 2, 2)
const X = CQ[0 1; 1 0]
const Y = CQ[0 -im; im 0]
const Z = CQ[1 0; 0 -1]

function density_between_endpoints(d)
    # A pure density matrix written directly, without a normalized vector's sqrt(2).
    rho = zeros(CQ, d, d)
    for i in (1, d), j in (1, d)
        rho[i, j] = q(1, 2)
    end
    @assert rho == rho'
    @assert rho * rho == rho
    @assert tr(rho) == 1
    return rho
end

function projectors(observable)
    @assert observable == observable'
    @assert observable * observable == I2
    effects = ((I2 + observable) * q(1, 2), (I2 - observable) * q(1, 2))
    @assert effects[1] + effects[2] == I2
    for i in 1:2, j in 1:2
        @assert effects[i] == effects[i]'
        @assert effects[i] * effects[j] == (i == j ? effects[i] : zeros(CQ, 2, 2))
    end
    return effects
end

function probability(rho, effects)
    value = tr(rho * reduce(kron, effects))
    @assert iszero(imag(value))
    @assert real(value) >= 0
    return real(value)
end

function check_chsh()
    # A rational point on the unit circle near (1/sqrt(2), 1/sqrt(2)).
    t = q(408, 985)
    c = (1 - t^2) / (1 + t^2)
    s = 2t / (1 + t^2)
    @assert c == q(803761, 1136689)
    @assert s == q(803760, 1136689)
    @assert c^2 + s^2 == 1
    alice = (projectors(Z), projectors(X))
    bob = (projectors(c * Z + s * X), projectors(c * Z - s * X))
    rho = density_between_endpoints(4)
    score = q(0)
    for x in 0:1, y in 0:1
        normalization = q(0)
        for a in 0:1, b in 0:1
            p = probability(rho, (alice[x+1][a+1], bob[y+1][b+1]))
            normalization += p
            if xor(a, b) == x*y
                score += p * q(1, 4)
            end
        end
        @assert normalization == 1
    end
    expected = q(3880899, 4546756)
    @assert score == expected
    @assert score == q(1, 2) + (c + s)/4
    @assert score > q(3, 4)
    return score
end

function check_ghz()
    party = (projectors(X), projectors(Y))
    rho = density_between_endpoints(8)
    score = q(0)
    for inputs in Iterators.product(0:1, 0:1, 0:1)
        normalization = q(0)
        promised = iseven(sum(inputs))
        for outputs in Iterators.product(0:1, 0:1, 0:1)
            effects = ntuple(i -> party[inputs[i]+1][outputs[i]+1], 3)
            p = probability(rho, effects)
            normalization += p
            wins = sum(outputs) % 2 == div(sum(inputs), 2) % 2
            if promised && wins
                score += p * q(1, 4)
            elseif promised
                @assert p == 0
            end
        end
        @assert normalization == 1
    end
    @assert score == 1
    return score
end

function check_magic_square()
    unit = Matrix{CQ}(I, 4, 4)
    grid = ((kron(X, I2), kron(I2, X), kron(X, X)),
            (kron(I2, Z), kron(Z, I2), kron(Z, Z)),
            (-kron(X, Z), -kron(Z, X), kron(Y, Y)))
    for r in 1:3
        @assert reduce(*, grid[r]) == unit
    end
    for c in 1:3
        @assert reduce(*, ntuple(r -> grid[r][c], 3)) == -unit
    end
    for group in (grid..., (ntuple(r -> grid[r][c], 3) for c in 1:3)...)
        for left in group, right in group
            @assert left == left'
            @assert left * left == unit
            @assert left * right == right * left
        end
    end
    even = ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0))
    odd = ((0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1))
    function joint_projectors(observables, outputs)
        effects = Tuple(reduce(*, ntuple(i -> (unit + (-1)^bits[i] * observables[i]) * q(1, 2), 3))
                        for bits in outputs)
        @assert reduce(+, effects) == unit
        for i in eachindex(effects), j in eachindex(effects)
            @assert effects[i] == effects[i]'
            @assert effects[i] * effects[j] == (i == j ? effects[i] : zeros(CQ, 4, 4))
        end
        return effects
    end
    alice = ntuple(r -> joint_projectors(grid[r], even), 3)
    bob = ntuple(c -> joint_projectors(ntuple(r -> transpose(grid[r][c]), 3), odd), 3)
    rho = zeros(CQ, 16, 16)
    for a in 1:4, b in 1:4
        rho[4*(a-1)+a, 4*(b-1)+b] = q(1, 4)
    end
    @assert rho == rho' && rho * rho == rho && tr(rho) == 1
    score = q(0)
    for r in 1:3, c in 1:3
        normalization = q(0)
        for a in 1:4, b in 1:4
            p = probability(rho, (alice[r][a], bob[c][b]))
            normalization += p
            if even[a][c] == odd[b][r]
                score += p * q(1, 9)
            else
                @assert p == 0
            end
        end
        @assert normalization == 1
    end
    @assert score == 1
    return score
end

println("CHSH rational feasible score = ", check_chsh())
println("GHZ exact reference score = ", check_ghz())
println("Magic-square exact reference score = ", check_magic_square())
println("Verified exact density, projectivity, Born probabilities, and normalization.")
println("This verifies three fixed references; it is not a general optimizer or adapter.")
