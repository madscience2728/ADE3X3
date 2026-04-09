module SetupSystem

using LinearAlgebra
using Random
using HomotopyContinuation

if !isdefined(Main, :OrbitData)
    include("orbit_data.jl")
end

export EDProblem, build_actual_target, build_ed_problem, build_generic_target, build_matmul_tensor,
       chart_tensor_numeric, unsupported_reason

struct EDProblem
    rank::Int
    orbit_config::Vector{Int}
    eff_params::Int
    variables
    parameters
    system::System
end

const S3 = [
    (1, 2, 3),
    (1, 3, 2),
    (2, 1, 3),
    (2, 3, 1),
    (3, 1, 2),
    (3, 2, 1),
]

const GROUP = [
    (pi, (e1, e2, e3)) for pi in S3 for e1 in (false, true) for e2 in (false, true)
    for e3 in (false, true)
]

const ORBIT_REPRESENTATIVES = Dict(
    1 => (0, 0, 0),
    6 => (0, 0, 1),
    12 => (0, 1, 1),
    8 => (1, 1, 1),
)

const ORBIT_PARAM_DIMS = Dict(1 => 3, 6 => 9, 12 => 12, 8 => 6)

swap12_value(x::Int) = x == 0 ? 0 : 3 - x

swap_perm(flag::Bool) = flag ? (1, 3, 2) : (1, 2, 3)

function apply_triple(pi, eps, triple::NTuple{3, Int})
    x = ntuple(i -> triple[pi[i]], 3)
    return ntuple(i -> eps[i] ? swap12_value(x[i]) : x[i], 3)
end

function perm_matrix(flag::Bool)
    P = zeros(Int, 3, 3)
    perm = swap_perm(flag)
    for i in 1:3
        P[i, perm[i]] = 1
    end
    return P
end

function permute_rows(M, perm)
    return M[collect(perm), :]
end

function permute_cols(M, perm)
    return M[:, collect(perm)]
end

function apply_factors(pi, eps, A, B, C)
    P = ntuple(i -> swap_perm(eps[i]), 3)
    pi_inv = Vector{Int}(undef, 3)
    for i in 1:3
        pi_inv[pi[i]] = i
    end
    facs = Dict((1, 2) => A, (2, 3) => B, (1, 3) => C)
    out = Dict{Symbol, Any}()
    for ((x, y), name) in (((1, 2), :A), ((2, 3), :B), ((1, 3), :C))
        ox, oy = pi_inv[x], pi_inv[y]
        key = (min(ox, oy), max(ox, oy))
        F = (ox, oy) == key ? facs[key] : transpose(facs[key])
        out[name] = permute_cols(permute_rows(F, P[x]), P[y])
    end
    return out[:A], out[:B], out[:C]
end

function orbit_transporters(orbit_size::Int)
    rep = ORBIT_REPRESENTATIVES[orbit_size]
    seen = Set{NTuple{3, Int}}()
    out = Vector{Tuple{Tuple{Int, Int, Int}, Tuple{Bool, Bool, Bool}, NTuple{3, Int}}}()
    for (pi, eps) in GROUP
        t = apply_triple(pi, eps, rep)
        if !(t in seen)
            push!(seen, t)
            push!(out, (pi, eps, t))
        end
    end
    return out
end

const ORBIT_TRANSPORTERS = Dict(size => orbit_transporters(size) for size in keys(ORBIT_REPRESENTATIVES))

function corner_seed(params)
    a0, a1, a2 = params
    M = Any[a0 a1 a1; a1 a2 a2; a1 a2 a2]
    return M, M, M
end

function edge_seed(params)
    a0, a1, a2, b00, b01, b02, b10, b11, b12 = params
    A = Any[a0 a1 a1; a1 a2 a2; a1 a2 a2]
    BC = Any[b00 b01 b02; b10 b11 b12; b10 b11 b12]
    return A, BC, BC
end

function face_seed(params)
    a00, a01, a02, a10, a11, a12, b00, b01, b02, b11, b12, b22 = params
    AC = Any[a00 a01 a02; a10 a11 a12; a10 a11 a12]
    B = Any[b00 b01 b02; b01 b11 b12; b02 b12 b22]
    return AC, B, AC
end

function interior_seed(params)
    u00, u01, u02, u11, u12, u22 = params
    M = Any[u00 u01 u02; u01 u11 u12; u02 u12 u22]
    return M, M, M
end

function seed_matrices(orbit_size::Int, params)
    orbit_size == 1 && return corner_seed(params)
    orbit_size == 6 && return edge_seed(params)
    orbit_size == 12 && return face_seed(params)
    orbit_size == 8 && return interior_seed(params)
    error("Unsupported orbit size $(orbit_size)")
end

function empty_tensor_vector(zero_value)
    return [zero_value for _ in 1:729]
end

function add_rank1_tensor!(acc, A, B, C)
    idx = 1
    a_flat = vec(A)
    b_flat = vec(B)
    c_flat = vec(C)
    for k in 1:9
        for j in 1:9
            for i in 1:9
                acc[idx] += a_flat[i] * b_flat[j] * c_flat[k]
                idx += 1
            end
        end
    end
    return acc
end

function chart_tensor_expressions(vars, orbit_config::Vector{Int})
    zero_value = zero(vars[1])
    tensor_entries = empty_tensor_vector(zero_value)
    offset = 1
    for orbit_size in orbit_config
        count = ORBIT_PARAM_DIMS[orbit_size]
        params = vars[offset:(offset + count - 1)]
        A_seed, B_seed, C_seed = seed_matrices(orbit_size, params)
        for (pi, eps, _) in ORBIT_TRANSPORTERS[orbit_size]
            A_term, B_term, C_term = apply_factors(pi, eps, A_seed, B_seed, C_seed)
            add_rank1_tensor!(tensor_entries, A_term, B_term, C_term)
        end
        offset += count
    end
    return tensor_entries
end

function chart_tensor_numeric(params::Vector{Float64}, orbit_config::Vector{Int})
    tensor_entries = zeros(Float64, 729)
    offset = 1
    for orbit_size in orbit_config
        count = ORBIT_PARAM_DIMS[orbit_size]
        seed_params = params[offset:(offset + count - 1)]
        A_seed, B_seed, C_seed = seed_matrices(orbit_size, seed_params)
        for (pi, eps, _) in ORBIT_TRANSPORTERS[orbit_size]
            A_term, B_term, C_term = apply_factors(pi, eps, A_seed, B_seed, C_seed)
            add_rank1_tensor!(tensor_entries, A_term, B_term, C_term)
        end
        offset += count
    end
    return reshape(tensor_entries, 9, 9, 9)
end

function build_actual_target()
    return vec(build_matmul_tensor())
end

function build_ed_problem(rank::Int)
    spec = Main.OrbitData.rank_spec(rank)
    spec.trivial_placeholder && error("Rank $(rank) is a placeholder case and has no nontrivial ED system.")

    nvars = spec.eff_params
    @var x[1:nvars] p[1:729]
    tensor_entries = chart_tensor_expressions(collect(x), spec.orbit_config)
    objective = sum((tensor_entries[i] - p[i])^2 for i in 1:729)
    equations = [differentiate(objective, x[i]) for i in 1:nvars]
    system = System(equations; variables = x, parameters = p)
    return EDProblem(rank, spec.orbit_config, spec.eff_params, x, p, system)
end

function build_matmul_tensor()
    tensor = zeros(Float64, 9, 9, 9)
    for i in 0:2
        for j in 0:2
            for k in 0:2
                tensor[i * 3 + j + 1, j * 3 + k + 1, i * 3 + k + 1] = 1.0
            end
        end
    end
    return tensor
end

function build_generic_target(seed::Int)
    rng = MersenneTwister(seed)
    return randn(rng, 9, 9, 9)
end

function unsupported_reason(spec::Main.OrbitData.RankSpec)
    return join([
        spec.note,
        "The one-orbit symmetric chart is now fixed by representation theory: nontrivial stabilizer isotypic components annihilate under orbit averaging.",
        "This scaffold currently attempts an actual HomotopyContinuation solve only for R13, which is the requested first validation case."
    ], " ")
end

end