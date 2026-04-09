include("orbit_data.jl")
include("setup_system.jl")

using JSON
using LinearAlgebra
using HomotopyContinuation

struct Candidate
    index::Int
    rank::Int
    orbit_config::Vector{Int}
    z::Vector{ComplexF64}
    imag_norm::Float64
    frob_error::Float64
end

function parse_complex_vector(parameters)
    return [ComplexF64(Float64(entry["re"]), Float64(entry["im"])) for entry in parameters]
end

function read_candidates(path::AbstractString; limit::Int=13)
    out = Candidate[]
    target = Main.SetupSystem.build_matmul_tensor()
    open(path, "r") do io
        for (line_index, line) in enumerate(eachline(io))
            line_index > limit && break
            isempty(strip(line)) && continue
            payload = JSON.parse(line)
            z = parse_complex_vector(payload["parameters"])
            orbit_config = Int[payload_orbit for payload_orbit in payload["orbit_config"]]
            real_params = Float64[real(value) for value in z]
            approx = Main.SetupSystem.chart_tensor_numeric(real_params, orbit_config)
            frob_error = norm(target - approx)
            imag_norm = norm(Float64[imag(value) for value in z])
            push!(out, Candidate(line_index, Int(payload["rank"]), orbit_config, z, imag_norm, frob_error))
        end
    end
    return out
end

function newton_correct(system, target_parameters::Vector{ComplexF64}, start::Vector{ComplexF64};
    tol::Float64=1e-8,
    max_iters::Int=100,
    singularity_tol::Float64=1e-14,
)
    x = copy(start)
    last_sigma_min = NaN
    for iter in 1:max_iters
        residual = ComplexF64.(evaluate(system, x, target_parameters))
        res_norm = norm(residual)
        J = ComplexF64.(jacobian(system, x, target_parameters))
        sigma_min = minimum(svdvals(Matrix(J)))
        last_sigma_min = sigma_min
        if res_norm <= tol
            return Dict(
                "converged" => true,
                "reason" => "converged",
                "iterations" => iter - 1,
                "residual_norm" => res_norm,
                "sigma_min" => sigma_min,
                "x" => x,
            )
        end
        if !isfinite(sigma_min) || sigma_min < singularity_tol
            return Dict(
                "converged" => false,
                "reason" => "singular_jacobian",
                "iterations" => iter - 1,
                "residual_norm" => res_norm,
                "sigma_min" => sigma_min,
                "x" => x,
            )
        end
        delta = -(J \ residual)
        if any(!isfinite(real(value)) || !isfinite(imag(value)) for value in delta)
            return Dict(
                "converged" => false,
                "reason" => "nonfinite_step",
                "iterations" => iter - 1,
                "residual_norm" => res_norm,
                "sigma_min" => sigma_min,
                "x" => x,
            )
        end
        x .= x .+ delta
        if norm(delta) <= tol && res_norm <= sqrt(tol)
            residual = ComplexF64.(evaluate(system, x, target_parameters))
            J = ComplexF64.(jacobian(system, x, target_parameters))
            sigma_min = minimum(svdvals(Matrix(J)))
            return Dict(
                "converged" => norm(residual) <= sqrt(tol),
                "reason" => norm(residual) <= sqrt(tol) ? "converged" : "stalled",
                "iterations" => iter,
                "residual_norm" => norm(residual),
                "sigma_min" => sigma_min,
                "x" => x,
            )
        end
    end
    residual = ComplexF64.(evaluate(system, x, target_parameters))
    return Dict(
        "converged" => false,
        "reason" => "max_iters",
        "iterations" => max_iters,
        "residual_norm" => norm(residual),
        "sigma_min" => last_sigma_min,
        "x" => x,
    )
end

function straight_line_homotopy(candidate::Candidate; step_size::Float64=0.05)
    problem = Main.SetupSystem.build_ed_problem(candidate.rank)
    target_parameters = ComplexF64.(Main.SetupSystem.build_actual_target())
    z0 = candidate.z
    x_prev = copy(z0)
    steps = Vector{Dict{String, Any}}()
    for theta in 0.0:step_size:1.0
        predictor = ComplexF64.(real.(z0), (1.0 - theta) .* imag.(z0))
        start = theta == 0.0 ? z0 : x_prev .+ ComplexF64.(zeros(length(z0)), -step_size .* imag.(z0))
        correction = newton_correct(problem.system, target_parameters, start)
        x_corr = correction["x"]
        imag_norm = norm(Float64[imag(value) for value in x_corr])
        push!(steps, Dict(
            "theta" => round(theta; digits=2),
            "predictor_imag_norm" => norm(Float64[imag(value) for value in predictor]),
            "corrected_imag_norm" => imag_norm,
            "residual_norm" => correction["residual_norm"],
            "sigma_min" => correction["sigma_min"],
            "reason" => correction["reason"],
            "converged" => correction["converged"],
        ))
        if !correction["converged"]
            return Dict(
                "status" => "singularity_or_failure",
                "failed_theta" => round(theta; digits=2),
                "reason" => correction["reason"],
                "steps" => steps,
            )
        end
        x_prev = x_corr
    end
    final_imag_norm = norm(Float64[imag(value) for value in x_prev])
    return Dict(
        "status" => final_imag_norm <= 1e-6 ? "reached_theta_1_real_solution" : "reached_theta_1_not_real",
        "final_imag_norm" => final_imag_norm,
        "steps" => steps,
    )
end

function main(args)
    if isempty(args)
        error("usage: julia analyze_complex_stream.jl <complex_solutions.jsonl> [limit]")
    end
    path = args[1]
    limit = length(args) >= 2 ? parse(Int, args[2]) : 13
    candidates = read_candidates(path; limit = limit)
    isempty(candidates) && error("No candidates found in $(path)")

    println("Scoring Re(z) against T_matmul")
    for candidate in candidates
        println(JSON.json(Dict(
            "index" => candidate.index,
            "rank" => candidate.rank,
            "orbit_config" => candidate.orbit_config,
            "imag_norm" => candidate.imag_norm,
            "frobenius_error" => candidate.frob_error,
        )))
    end

    _, best_index = findmin(candidate -> candidate.frob_error, candidates)
    best_candidate = candidates[best_index]
    println()
    println("Best real-space estimate")
    println(JSON.json(Dict(
        "index" => best_candidate.index,
        "rank" => best_candidate.rank,
        "orbit_config" => best_candidate.orbit_config,
        "imag_norm" => best_candidate.imag_norm,
        "frobenius_error" => best_candidate.frob_error,
    )))

    println()
    println("Straight-line homotopy")
    homotopy = straight_line_homotopy(best_candidate)
    println(JSON.json(homotopy))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end