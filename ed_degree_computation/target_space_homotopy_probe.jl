include("target_perturbation_probe.jl")

using JSON
using LinearAlgebra

function build_perturbation_trials(candidate::Candidate; scale::Float64, trials::Int=12, seed::Int=20260409)
    problem, base_target, base_correction = corrected_base_point(candidate)
    base_point = base_correction["x"]
    base_imag_norm = norm(Float64[imag(value) for value in base_point])
    rng = MersenneTwister(seed)
    entries = Vector{Dict{String, Any}}()
    for trial in 1:trials
        perturbation = normalized_random_perturbation(rng, length(base_target), scale)
        perturbed_target = base_target .+ ComplexF64.(perturbation)
        correction = newton_correct(problem.system, perturbed_target, base_point)
        corrected = correction["x"]
        imag_norm = norm(Float64[imag(value) for value in corrected])
        push!(entries, Dict(
            "trial" => trial,
            "perturbation" => perturbation,
            "perturbed_target" => perturbed_target,
            "corrected_point" => corrected,
            "imag_norm" => imag_norm,
            "delta_imag" => imag_norm - base_imag_norm,
            "converged" => correction["converged"],
            "reason" => correction["reason"],
        ))
    end
    return problem, base_target, base_point, entries
end

function track_branch(problem, exact_target::Vector{ComplexF64}, perturbed_target::Vector{ComplexF64}, start_point::Vector{ComplexF64}; step_size::Float64=0.005)
    x_prev = copy(start_point)
    steps = Vector{Dict{String, Any}}()
    for s in 0.0:step_size:1.0
        target = s .* exact_target .+ (1.0 - s) .* perturbed_target
        correction = newton_correct(problem.system, target, x_prev)
        x_corr = correction["x"]
        imag_norm = norm(Float64[imag(value) for value in x_corr])
        sigma_min = minimum(svdvals(Matrix(ComplexF64.(jacobian(problem.system, x_corr, target)))))
        push!(steps, Dict(
            "s" => round(s; digits=3),
            "imag_norm" => imag_norm,
            "sigma_min" => sigma_min,
            "residual_norm" => norm(ComplexF64.(evaluate(problem.system, x_corr, target))),
            "converged" => correction["converged"],
            "reason" => correction["reason"],
            "z" => x_corr,
        ))
        correction["converged"] || break
        x_prev = x_corr
    end
    return steps
end

function serialize_point(point::Vector{ComplexF64})
    return [Dict("re" => Float64(real(value)), "im" => Float64(imag(value))) for value in point]
end

function find_branch_point(primary_steps::Vector{Dict{String, Any}}, conjugate_steps::Vector{Dict{String, Any}})
    count = min(length(primary_steps), length(conjugate_steps))
    count < 2 && error("Need at least two tracked steps to detect a branch point")
    deltas = Float64[]
    for idx in 1:(count - 1)
        current_imag = min(primary_steps[idx]["imag_norm"], conjugate_steps[idx]["imag_norm"])
        next_imag = min(primary_steps[idx + 1]["imag_norm"], conjugate_steps[idx + 1]["imag_norm"])
        push!(deltas, next_imag - current_imag)
    end
    _, max_index = findmax(deltas)
    return Dict(
        "index" => max_index,
        "s_star" => primary_steps[max_index]["s"],
        "s_next" => primary_steps[max_index + 1]["s"],
        "delta_imag" => deltas[max_index],
        "primary_step" => primary_steps[max_index],
        "conjugate_step" => conjugate_steps[max_index],
    )
end

function epsilon_at_branch(candidate::Candidate, branch_step::Dict{String, Any})
    real_params = Float64[real(value) for value in branch_step["z"]]
    approx = Main.SetupSystem.chart_tensor_numeric(real_params, candidate.orbit_config)
    target = Main.SetupSystem.build_matmul_tensor()
    return norm(target - approx)
end

function main(args)
    if length(args) < 2
        error("usage: julia target_space_homotopy_probe.jl <complex_solutions.jsonl> <index>")
    end
    path = args[1]
    requested_index = parse(Int, args[2])
    candidates = read_candidates(path; limit = requested_index)
    length(candidates) >= requested_index || error("Requested index $(requested_index) not found")
    candidate = candidates[requested_index]

    problem, exact_target, _, perturbation_trials = build_perturbation_trials(candidate; scale = 0.001)
    successful_trials = [trial for trial in perturbation_trials if trial["converged"]]
    isempty(successful_trials) && error("No successful scale=0.001 perturbation trials found")
    imag_norms = [trial["imag_norm"] for trial in successful_trials]
    _, best_trial_index = findmin(imag_norms)
    selected_trial = successful_trials[best_trial_index]

    primary_steps = track_branch(problem, exact_target, selected_trial["perturbed_target"], selected_trial["corrected_point"])
    conjugate_steps = track_branch(problem, exact_target, selected_trial["perturbed_target"], conj.(selected_trial["corrected_point"]))
    branch_point = find_branch_point(primary_steps, conjugate_steps)
    epsilon13 = epsilon_at_branch(candidate, branch_point["primary_step"])

    println(JSON.json(Dict(
        "event" => "selected_perturbed_target",
        "rank" => candidate.rank,
        "orbit_config" => candidate.orbit_config,
        "index" => candidate.index,
        "scale" => 0.001,
        "trial" => selected_trial["trial"],
        "initial_imag_norm" => selected_trial["imag_norm"],
    )))

    for idx in 1:min(length(primary_steps), length(conjugate_steps))
        println(JSON.json(Dict(
            "event" => "target_homotopy_step",
            "s" => primary_steps[idx]["s"],
            "primary_imag_norm" => primary_steps[idx]["imag_norm"],
            "conjugate_imag_norm" => conjugate_steps[idx]["imag_norm"],
            "primary_sigma_min" => primary_steps[idx]["sigma_min"],
            "conjugate_sigma_min" => conjugate_steps[idx]["sigma_min"],
        )))
    end

    println(JSON.json(Dict(
        "event" => "branch_point",
        "s_star" => branch_point["s_star"],
        "s_next" => branch_point["s_next"],
        "delta_imag" => branch_point["delta_imag"],
        "primary_imag_norm" => branch_point["primary_step"]["imag_norm"],
        "conjugate_imag_norm" => branch_point["conjugate_step"]["imag_norm"],
        "primary_sigma_min" => branch_point["primary_step"]["sigma_min"],
        "conjugate_sigma_min" => branch_point["conjugate_step"]["sigma_min"],
        "real_part_at_branch" => [Float64(real(value)) for value in branch_point["primary_step"]["z"]],
        "epsilon13" => epsilon13,
    )))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end