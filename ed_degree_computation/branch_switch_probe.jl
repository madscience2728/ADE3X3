include("analyze_complex_stream.jl")

using JSON
using LinearAlgebra

function complex_vector_payload(values)
    return [Dict("re" => Float64(real(value)), "im" => Float64(imag(value))) for value in values]
end

function step_diagnostics(problem, target_parameters::Vector{ComplexF64}, point::Vector{ComplexF64})
    residual = ComplexF64.(evaluate(problem.system, point, target_parameters))
    J = ComplexF64.(jacobian(problem.system, point, target_parameters))
    F = svd(Matrix(J))
    return Dict(
        "residual_norm" => norm(residual),
        "sigma_min" => F.S[end],
        "u_min" => F.U[:, end],
        "v_min" => F.V[:, end],
    )
end

function sweep_candidate(candidate::Candidate; step_size::Float64=0.05)
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
        diag = step_diagnostics(problem, target_parameters, x_corr)
        imag_norm = norm(Float64[imag(value) for value in x_corr])
        push!(steps, Dict(
            "theta" => round(theta; digits=2),
            "predictor_imag_norm" => norm(Float64[imag(value) for value in predictor]),
            "corrected_imag_norm" => imag_norm,
            "converged" => correction["converged"],
            "reason" => correction["reason"],
            "newton_residual_norm" => correction["residual_norm"],
            "residual_norm" => diag["residual_norm"],
            "sigma_min" => diag["sigma_min"],
            "z" => x_corr,
            "u_min" => diag["u_min"],
            "v_min" => diag["v_min"],
        ))
        correction["converged"] || break
        x_prev = x_corr
    end
    return steps
end

function continue_from_step(candidate::Candidate, start_theta::Float64, start_point::Vector{ComplexF64}; step_size::Float64=0.05)
    problem = Main.SetupSystem.build_ed_problem(candidate.rank)
    target_parameters = ComplexF64.(Main.SetupSystem.build_actual_target())
    x_prev = copy(start_point)
    steps = Vector{Dict{String, Any}}()
    theta = start_theta + step_size
    while theta <= 1.0 + 1e-9
        start = x_prev .+ ComplexF64.(zeros(length(x_prev)), -step_size .* imag.(candidate.z))
        correction = newton_correct(problem.system, target_parameters, start)
        x_corr = correction["x"]
        diag = step_diagnostics(problem, target_parameters, x_corr)
        imag_norm = norm(Float64[imag(value) for value in x_corr])
        push!(steps, Dict(
            "theta" => round(theta; digits=2),
            "corrected_imag_norm" => imag_norm,
            "converged" => correction["converged"],
            "reason" => correction["reason"],
            "residual_norm" => diag["residual_norm"],
            "sigma_min" => diag["sigma_min"],
            "z" => x_corr,
        ))
        correction["converged"] || return Dict(
            "status" => "failed",
            "failed_theta" => round(theta; digits=2),
            "reason" => correction["reason"],
            "steps" => steps,
        )
        x_prev = x_corr
        theta += step_size
    end
    return Dict(
        "status" => norm(Float64[imag(value) for value in x_prev]) <= 1e-6 ? "reached_theta_1_real_solution" : "reached_theta_1_not_real",
        "final_imag_norm" => norm(Float64[imag(value) for value in x_prev]),
        "steps" => steps,
    )
end

function probe_switch(candidate::Candidate, branch_step::Dict{String, Any}, deltas::Vector{Float64})
    problem = Main.SetupSystem.build_ed_problem(candidate.rank)
    target_parameters = ComplexF64.(Main.SetupSystem.build_actual_target())
    original_z = branch_step["z"]
    original_imag_norm = branch_step["corrected_imag_norm"]
    u_min = branch_step["u_min"]
    probes = Vector{Dict{String, Any}}()
    successes = Vector{Dict{String, Any}}()
    for delta in deltas
        switched_start = original_z .+ ComplexF64(delta) .* u_min
        correction = newton_correct(problem.system, target_parameters, switched_start)
        corrected = correction["x"]
        diag = step_diagnostics(problem, target_parameters, corrected)
        imag_norm = norm(Float64[imag(value) for value in corrected])
        improved = correction["converged"] && imag_norm < original_imag_norm
        payload = Dict(
            "delta" => delta,
            "converged" => correction["converged"],
            "reason" => correction["reason"],
            "sigma_min" => diag["sigma_min"],
            "residual_norm" => diag["residual_norm"],
            "imag_norm" => imag_norm,
            "improvement" => original_imag_norm - imag_norm,
            "switched" => improved,
            "z" => corrected,
        )
        push!(probes, payload)
        improved && push!(successes, payload)
    end
    return probes, successes
end

function serialize_step(step::Dict{String, Any})
    return Dict(
        "theta" => step["theta"],
        "corrected_imag_norm" => step["corrected_imag_norm"],
        "sigma_min" => step["sigma_min"],
        "residual_norm" => step["residual_norm"],
        "converged" => step["converged"],
        "reason" => step["reason"],
        "z" => complex_vector_payload(step["z"]),
        "u_min" => haskey(step, "u_min") ? complex_vector_payload(step["u_min"]) : Any[],
    )
end

function main(args)
    if length(args) < 2
        error("usage: julia branch_switch_probe.jl <complex_solutions.jsonl> <index>")
    end
    path = args[1]
    requested_index = parse(Int, args[2])
    candidates = read_candidates(path; limit = requested_index)
    length(candidates) >= requested_index || error("Requested index $(requested_index) not found")
    candidate = candidates[requested_index]

    steps = sweep_candidate(candidate)
    for step in steps
        println(JSON.json(Dict(
            "event" => "theta_step",
            "theta" => step["theta"],
            "rank" => candidate.rank,
            "orbit_config" => candidate.orbit_config,
            "sigma_min" => step["sigma_min"],
            "corrected_imag_norm" => step["corrected_imag_norm"],
            "z" => complex_vector_payload(step["z"]),
            "u_min" => complex_vector_payload(step["u_min"]),
        )))
    end

    sigma_values = [Float64(step["sigma_min"]) for step in steps]
    _, min_step_index = findmin(sigma_values)
    critical_step = steps[min_step_index]
    println(JSON.json(Dict(
        "event" => "most_ill_conditioned_step",
        "theta" => critical_step["theta"],
        "sigma_min" => critical_step["sigma_min"],
        "corrected_imag_norm" => critical_step["corrected_imag_norm"],
    )))

    deltas = [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1]
    probes, successes = probe_switch(candidate, critical_step, deltas)
    for probe in probes
        println(JSON.json(Dict(
            "event" => "switch_probe",
            "theta" => critical_step["theta"],
            "delta" => probe["delta"],
            "converged" => probe["converged"],
            "reason" => probe["reason"],
            "sigma_min" => probe["sigma_min"],
            "imag_norm" => probe["imag_norm"],
            "improvement" => probe["improvement"],
            "switched" => probe["switched"],
        )))
    end

    if isempty(successes)
        println(JSON.json(Dict(
            "event" => "branch_switch_result",
            "success" => false,
            "theta" => critical_step["theta"],
            "message" => "No delta along requested u_min direction reduced |Im| at the most ill-conditioned step.",
        )))
        return
    end

    improvements = [probe["imag_norm"] for probe in successes]
    _, best_probe_index = findmin(improvements)
    best_probe = successes[best_probe_index]
    continuation = continue_from_step(candidate, Float64(critical_step["theta"]), best_probe["z"])
    println(JSON.json(Dict(
        "event" => "branch_switch_result",
        "success" => true,
        "theta" => critical_step["theta"],
        "delta" => best_probe["delta"],
        "switched_imag_norm" => best_probe["imag_norm"],
        "original_imag_norm" => critical_step["corrected_imag_norm"],
        "continuation_status" => continuation["status"],
        "final_imag_norm" => get(continuation, "final_imag_norm", nothing),
    )))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end