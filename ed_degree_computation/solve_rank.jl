module SolveRank

using JSON
using LinearAlgebra
using Random
using HomotopyContinuation

export solve_rank, write_result

function phase_log(enabled::Bool, rank::Int, start_time::Float64, message::AbstractString)
    enabled || return
    elapsed = round(time() - start_time; digits=3)
    println("[R=$(rank) +$(elapsed)s] $(message)")
    flush(stdout)
end

function trivial_payload(spec::Main.OrbitData.RankSpec)
    return Dict(
        "rank" => spec.rank,
        "eff_params" => spec.eff_params,
        "orbit_config" => spec.orbit_config,
        "ed_degree" => 1,
        "real_solutions" => 1,
        "best_frobenius" => 0.0,
        "best_solution" => Float64[],
        "wall_time_seconds" => 0.0,
        "status" => "placeholder",
        "note" => spec.note,
    )
end

function blocked_payload(spec::Main.OrbitData.RankSpec)
    return Dict(
        "rank" => spec.rank,
        "eff_params" => spec.eff_params,
        "orbit_config" => spec.orbit_config,
        "ed_degree" => nothing,
        "real_solutions" => nothing,
        "best_frobenius" => nothing,
        "best_solution" => Float64[],
        "wall_time_seconds" => 0.0,
        "status" => "pending",
        "note" => Main.SetupSystem.unsupported_reason(spec),
        "generic_h_rank" => spec.generic_h_rank,
        "target_h_rank" => spec.target_h_rank,
    )
end

function monodromy_payload(spec::Main.OrbitData.RankSpec, monodromy_result, wall_time_seconds::Float64, timings::Dict{String, Float64})
    return Dict(
        "rank" => spec.rank,
        "eff_params" => spec.eff_params,
        "orbit_config" => spec.orbit_config,
        "ed_degree" => nsolutions(monodromy_result),
        "real_solutions" => nothing,
        "best_frobenius" => nothing,
        "best_solution" => Float64[],
        "wall_time_seconds" => wall_time_seconds,
        "timings" => copy(timings),
        "status" => "monodromy_complete",
        "note" => "Monodromy finished and ED degree was recorded. Tracking to the actual tensor is still in progress.",
        "generic_h_rank" => spec.generic_h_rank,
        "target_h_rank" => spec.target_h_rank,
    )
end

function tracking_payload(spec::Main.OrbitData.RankSpec, monodromy_result, wall_time_seconds::Float64, timings::Dict{String, Float64})
    return Dict(
        "rank" => spec.rank,
        "eff_params" => spec.eff_params,
        "orbit_config" => spec.orbit_config,
        "ed_degree" => nsolutions(monodromy_result),
        "real_solutions" => nothing,
        "best_frobenius" => nothing,
        "best_solution" => Float64[],
        "wall_time_seconds" => wall_time_seconds,
        "timings" => copy(timings),
        "status" => "tracking_started",
        "note" => "Monodromy finished and target tracking has started. The solve is currently inside the blocking path-tracking call.",
        "generic_h_rank" => spec.generic_h_rank,
        "target_h_rank" => spec.target_h_rank,
    )
end

function persist_payload(payload::Dict{String, Any})
    path = Main.OrbitData.result_path(payload["rank"])
    open(path, "w") do io
        JSON.print(io, payload, 2)
    end
    return path
end

function solution_dump_path(rank::Int, label::AbstractString)
    return joinpath(@__DIR__, "results", "ed_R$(rank)_$(label).json")
end

function complex_entry(value)
    return Dict(
        "re" => Float64(real(value)),
        "im" => Float64(imag(value)),
    )
end

function solution_entry(solution)
    return [complex_entry(value) for value in solution]
end

function write_solution_dump(rank::Int, label::AbstractString, payload::Dict{String, Any})
    path = solution_dump_path(rank, label)
    open(path, "w") do io
        JSON.print(io, payload, 2)
    end
    return path
end

function session_jsonl_base_path()
    return get(ENV, "ADE_JSONL_PATH", joinpath(@__DIR__, "results", "session_live.jsonl"))
end

function solution_stream_path(label::AbstractString)
    base_path = session_jsonl_base_path()
    stem, extension = splitext(base_path)
    return string(stem, "_", label, extension)
end

function counts_path(rank::Int)
    return joinpath(@__DIR__, "results", "counts_R$(rank).txt")
end

function append_jsonl(path::AbstractString, payload::Dict{String, Any})
    open(path, "a") do io
        JSON.print(io, payload)
        write(io, '\n')
    end
    return path
end

function write_counts_file(rank::Int, total_found::Int, real_found::Int, start_time::Float64)
    path = counts_path(rank)
    open(path, "w") do io
        write(io, "total_found=$(total_found)\n")
        write(io, "real_found=$(real_found)\n")
        write(io, "elapsed_seconds=$(round(time() - start_time; digits=6))\n")
        write(io, "updated_at=$(time())\n")
    end
    return path
end

function solution_signature(solution)
    return join(
        (
            string(round(Float64(real(value)); digits=12), ",", round(Float64(imag(value)); digits=12))
            for value in solution
        ),
        ";",
    )
end

function stream_monodromy_solutions!(
    rank::Int,
    orbit_config,
    seen_signatures::Set{String},
    real_signatures::Set{String},
    complex_signatures::Set{String},
    loop_results,
    start_time::Float64,
    last_counts_write::Base.RefValue{Float64},
)
    real_jsonl_path = solution_stream_path("real_solutions")
    complex_jsonl_path = solution_stream_path("complex_solutions")
    for result in loop_results
        is_success(result) || continue
        current_solution = solution(result)
        signature = solution_signature(current_solution)
        signature in seen_signatures && continue
        push!(seen_signatures, signature)
        if is_real_solution(current_solution)
            signature in real_signatures && continue
            push!(real_signatures, signature)
            payload = Dict(
                "event" => "monodromy_solution",
                "rank" => rank,
                "orbit_config" => orbit_config,
                "timestamp" => time(),
                "elapsed_seconds" => round(time() - start_time; digits=6),
                "solution_index" => length(real_signatures),
                "residual" => Float64(residual(result)),
                "accuracy" => Float64(accuracy(result)),
                "parameters" => solution_entry(current_solution),
                "real_parameters" => numeric_solution_vector(current_solution),
                "is_real" => true,
                "total_found" => length(seen_signatures),
                "real_found" => length(real_signatures),
            )
            append_jsonl(real_jsonl_path, payload)
        else
            signature in complex_signatures && continue
            push!(complex_signatures, signature)
            payload = Dict(
                "event" => "monodromy_solution",
                "rank" => rank,
                "orbit_config" => orbit_config,
                "timestamp" => time(),
                "elapsed_seconds" => round(time() - start_time; digits=6),
                "solution_index" => length(complex_signatures),
                "residual" => Float64(residual(result)),
                "accuracy" => Float64(accuracy(result)),
                "parameters" => solution_entry(current_solution),
                "real_parameters" => numeric_solution_vector(current_solution),
                "is_real" => false,
                "total_found" => length(seen_signatures),
                "real_found" => length(real_signatures),
            )
            append_jsonl(complex_jsonl_path, payload)
        end
    end
    now = time()
    if now - last_counts_write[] >= 60.0
        write_counts_file(rank, length(seen_signatures), length(real_signatures), start_time)
        last_counts_write[] = now
    end
    return false
end

function numeric_solution_vector(solution)
    return [Float64(real(value)) for value in solution]
end

function is_real_solution(solution; atol::Float64=1e-6)
    return maximum(abs(imag(value)) for value in solution) <= atol
end

function best_real_payload(solutions_vec, orbit_config)
    target = Main.SetupSystem.build_matmul_tensor()
    real_solutions = [solution for solution in solutions_vec if is_real_solution(solution)]
    if isempty(real_solutions)
        return 0, nothing, nothing
    end

    best_solution = nothing
    best_frob = Inf
    for solution in real_solutions
        numeric = numeric_solution_vector(solution)
        tensor = Main.SetupSystem.chart_tensor_numeric(numeric, orbit_config)
        frob = norm(tensor - target)
        if frob < best_frob
            best_frob = frob
            best_solution = numeric
        end
    end
    return length(real_solutions), best_frob, best_solution
end

function solved_payload(spec::Main.OrbitData.RankSpec, monodromy_result, actual_result, wall_time_seconds::Float64, timings::Dict{String, Float64})
    generic_count = nsolutions(monodromy_result)
    actual_solutions = solutions(actual_result)
    real_count, best_frob, best_solution = best_real_payload(actual_solutions, spec.orbit_config)
    real_dump_path = write_solution_dump(
        spec.rank,
        "tracked_real_solutions",
        Dict(
            "rank" => spec.rank,
            "orbit_config" => spec.orbit_config,
            "count" => real_count,
            "solutions" => [
                Dict(
                    "parameters" => solution_entry(solution),
                    "real_parameters" => numeric_solution_vector(solution),
                    "frobenius_to_target" => begin
                        tensor = Main.SetupSystem.chart_tensor_numeric(numeric_solution_vector(solution), spec.orbit_config)
                        Float64(norm(tensor - Main.SetupSystem.build_matmul_tensor()))
                    end,
                )
                for solution in actual_solutions if is_real_solution(solution)
            ],
        ),
    )
    return Dict(
        "rank" => spec.rank,
        "eff_params" => spec.eff_params,
        "orbit_config" => spec.orbit_config,
        "ed_degree" => generic_count,
        "real_solutions" => real_count,
        "best_frobenius" => best_frob,
        "best_solution" => best_solution === nothing ? Float64[] : best_solution,
        "wall_time_seconds" => wall_time_seconds,
        "timings" => timings,
        "status" => "solved",
        "note" => "Monodromy solve on the one-orbit symmetric chart, then tracked to the actual matmul tensor.",
        "tracked_real_solutions_path" => real_dump_path,
        "generic_h_rank" => spec.generic_h_rank,
        "target_h_rank" => spec.target_h_rank,
    )
end

function failed_payload(spec::Main.OrbitData.RankSpec, err, wall_time_seconds::Float64, timings::Dict{String, Float64})
    return Dict(
        "rank" => spec.rank,
        "eff_params" => spec.eff_params,
        "orbit_config" => spec.orbit_config,
        "ed_degree" => nothing,
        "real_solutions" => nothing,
        "best_frobenius" => nothing,
        "best_solution" => Float64[],
        "wall_time_seconds" => wall_time_seconds,
        "timings" => timings,
        "status" => "failed",
        "note" => string("R", spec.rank, " solve failed: ", err),
        "generic_h_rank" => spec.generic_h_rank,
        "target_h_rank" => spec.target_h_rank,
    )
end

function solve_rank(rank::Int; seed::Int=20260408, debug::Bool=false)
    spec = Main.OrbitData.rank_spec(rank)
    start_time = time()
    timings = Dict{String, Float64}()
    if spec.trivial_placeholder
        payload = trivial_payload(spec)
        payload["wall_time_seconds"] = round(time() - start_time; digits=6)
        payload["timings"] = timings
        return payload
    end

    phase_log(debug, rank, start_time, "building ED system")
    t0 = time()
    problem = Main.SetupSystem.build_ed_problem(rank)
    timings["build_system_seconds"] = round(time() - t0; digits=6)
    phase_log(debug, rank, start_time, string(
        "system ready: variables=", length(problem.variables),
        ", parameters=", length(problem.parameters),
        ", equations=", length(problem.variables)
    ))

    phase_log(debug, rank, start_time, "building actual target tensor")
    t0 = time()
    actual_target = Main.SetupSystem.build_actual_target()
    timings["build_target_seconds"] = round(time() - t0; digits=6)

    try
        phase_log(debug, rank, start_time, "finding start pair")
        t0 = time()
        start_pair = HomotopyContinuation.find_start_pair(problem.system; max_tries = 2_000)
        timings["find_start_pair_seconds"] = round(time() - t0; digits=6)
        start_pair === nothing && error("find_start_pair returned nothing")
        start_solution, start_parameters = start_pair
        phase_log(debug, rank, start_time, "start pair found")

        phase_log(debug, rank, start_time, "starting monodromy solve")
        t0 = time()
        seen_signatures = Set{String}()
        real_signatures = Set{String}()
        complex_signatures = Set{String}()
        last_counts_write = Ref(start_time)
        write_counts_file(rank, 0, 0, start_time)
        loop_callback = results -> stream_monodromy_solutions!(rank, spec.orbit_config, seen_signatures, real_signatures, complex_signatures, results, start_time, last_counts_write)
        monodromy_result = monodromy_solve(
            problem.system,
            [start_solution],
            start_parameters;
            seed = UInt32(seed + rank),
            show_progress = debug,
            threading = true,
            compile = false,
            warning = true,
            loop_finished_callback = loop_callback,
        )
        timings["monodromy_seconds"] = round(time() - t0; digits=6)
        write_counts_file(rank, length(seen_signatures), length(real_signatures), start_time)
        phase_log(debug, rank, start_time, string("monodromy finished with ", nsolutions(monodromy_result), " solutions"))
        monodromy_dump_path = write_solution_dump(
            rank,
            "monodromy_solutions",
            Dict(
                "rank" => rank,
                "orbit_config" => spec.orbit_config,
                "count" => nsolutions(monodromy_result),
                "solutions" => [solution_entry(solution) for solution in solutions(monodromy_result)],
            ),
        )
        monodromy_checkpoint = monodromy_payload(spec, monodromy_result, round(time() - start_time; digits=6), timings)
        monodromy_checkpoint["monodromy_solutions_path"] = monodromy_dump_path
        persist_payload(monodromy_checkpoint)
        phase_log(debug, rank, start_time, "monodromy checkpoint written")

        phase_log(debug, rank, start_time, "tracking solutions to actual tensor")
        tracking_checkpoint = tracking_payload(spec, monodromy_result, round(time() - start_time; digits=6), timings)
        tracking_checkpoint["monodromy_solutions_path"] = monodromy_dump_path
        persist_payload(tracking_checkpoint)
        phase_log(debug, rank, start_time, "tracking checkpoint written")
        t0 = time()
        actual_result = solve(
            problem.system,
            monodromy_result;
            target_parameters = ComplexF64.(actual_target),
            show_progress = debug,
            threading = true,
        )
        timings["track_actual_seconds"] = round(time() - t0; digits=6)
        phase_log(debug, rank, start_time, string("target tracking finished with ", length(solutions(actual_result)), " endpoints"))

        return solved_payload(spec, monodromy_result, actual_result, round(time() - start_time; digits=6), timings)
    catch err
        phase_log(debug, rank, start_time, string("solve failed: ", err))
        return failed_payload(spec, err, round(time() - start_time; digits=6), timings)
    end
end

function write_result(payload::Dict{String, Any})
    return persist_payload(payload)
end

end