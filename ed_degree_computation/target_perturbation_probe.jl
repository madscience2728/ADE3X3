include("analyze_complex_stream.jl")

using JSON
using LinearAlgebra
using Random

function corrected_base_point(candidate::Candidate)
    problem = Main.SetupSystem.build_ed_problem(candidate.rank)
    target_parameters = ComplexF64.(Main.SetupSystem.build_actual_target())
    correction = newton_correct(problem.system, target_parameters, candidate.z)
    return problem, target_parameters, correction
end

function normalized_random_perturbation(rng::AbstractRNG, n::Int, scale::Float64)
    direction = randn(rng, n)
    direction ./= norm(direction)
    return scale .* direction
end

function probe_scale(problem, base_target::Vector{ComplexF64}, base_point::Vector{ComplexF64}, base_imag_norm::Float64;
    scale::Float64,
    trials::Int,
    seed::Int,
)
    rng = MersenneTwister(seed)
    results = Vector{Dict{String, Any}}()
    for trial in 1:trials
        perturbation = normalized_random_perturbation(rng, length(base_target), scale)
        perturbed_target = base_target .+ ComplexF64.(perturbation)
        correction = newton_correct(problem.system, perturbed_target, base_point)
        corrected = correction["x"]
        imag_norm = norm(Float64[imag(value) for value in corrected])
        delta_imag = imag_norm - base_imag_norm
        push!(results, Dict(
            "trial" => trial,
            "scale" => scale,
            "converged" => correction["converged"],
            "reason" => correction["reason"],
            "residual_norm" => correction["residual_norm"],
            "imag_norm" => imag_norm,
            "delta_imag" => delta_imag,
            "moves_toward_real" => correction["converged"] && delta_imag < 0.0,
            "moves_away_from_real" => correction["converged"] && delta_imag > 0.0,
        ))
    end
    return results
end

function summarize_scale(scale_results::Vector{Dict{String, Any}})
    converged = [item for item in scale_results if item["converged"]]
    toward = [item for item in converged if item["moves_toward_real"]]
    away = [item for item in converged if item["moves_away_from_real"]]
    stable = [item for item in converged if !item["moves_toward_real"] && !item["moves_away_from_real"]]
    min_imag = isempty(converged) ? nothing : minimum(item["imag_norm"] for item in converged)
    max_imag = isempty(converged) ? nothing : maximum(item["imag_norm"] for item in converged)
    mean_delta = isempty(converged) ? nothing : sum(item["delta_imag"] for item in converged) / length(converged)
    return Dict(
        "scale" => scale_results[1]["scale"],
        "trials" => length(scale_results),
        "converged_trials" => length(converged),
        "toward_real_trials" => length(toward),
        "away_from_real_trials" => length(away),
        "stable_trials" => length(stable),
        "min_imag_norm" => min_imag,
        "max_imag_norm" => max_imag,
        "mean_delta_imag" => mean_delta,
    )
end

function overall_interpretation(summaries::Vector{Dict{String, Any}})
    toward_total = sum(summary["toward_real_trials"] for summary in summaries)
    converged_total = sum(summary["converged_trials"] for summary in summaries)
    if converged_total == 0
        return "No perturbed target converged, so this probe is inconclusive."
    end
    if toward_total == 0
        return "Across these perturbations, |Im| never decreased. This supports the component being genuinely complex rather than T_matmul merely missing a nearby real branch."
    end
    return "Some perturbed targets decreased |Im|. This supports a parametric effect: a less-complex or possibly real partner branch may exist near T_matmul in target space."
end

function main(args)
    if length(args) < 2
        error("usage: julia target_perturbation_probe.jl <complex_solutions.jsonl> <index> [trials]")
    end
    path = args[1]
    requested_index = parse(Int, args[2])
    trials = length(args) >= 3 ? parse(Int, args[3]) : 12
    candidates = read_candidates(path; limit = requested_index)
    length(candidates) >= requested_index || error("Requested index $(requested_index) not found")
    candidate = candidates[requested_index]

    problem, base_target, base_correction = corrected_base_point(candidate)
    base_point = base_correction["x"]
    base_imag_norm = norm(Float64[imag(value) for value in base_point])
    println(JSON.json(Dict(
        "event" => "base_point",
        "rank" => candidate.rank,
        "orbit_config" => candidate.orbit_config,
        "index" => candidate.index,
        "base_imag_norm" => base_imag_norm,
        "base_residual_norm" => base_correction["residual_norm"],
        "base_reason" => base_correction["reason"],
        "base_converged" => base_correction["converged"],
    )))

    scales = [0.001, 0.01, 0.1]
    summaries = Vector{Dict{String, Any}}()
    for (offset, scale) in enumerate(scales)
        scale_results = probe_scale(problem, base_target, base_point, base_imag_norm; scale = scale, trials = trials, seed = 20260408 + offset)
        for result in scale_results
            println(JSON.json(merge(Dict("event" => "perturbation_trial"), result)))
        end
        summary = summarize_scale(scale_results)
        push!(summaries, summary)
        println(JSON.json(merge(Dict("event" => "scale_summary"), summary)))
    end

    println(JSON.json(Dict(
        "event" => "overall_interpretation",
        "message" => overall_interpretation(summaries),
    )))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end