using JSON

include("orbit_data.jl")
include("setup_system.jl")
include("solve_rank.jl")

using .OrbitData
using .SolveRank

function format_cell(value)
    value === nothing && return "blocked"
    value isa AbstractFloat && return string(round(value; digits=6))
    return string(value)
end

function main()
    println("=== ED Degree Computation ===")
    println("Writing JSON outputs to results/")

    table_rows = Vector{Dict{String, Any}}()
    for rank in ALL_RANKS
        println("[R=$(rank)] starting")
        payload = solve_rank(rank)
        path = write_result(payload)
        status = payload["status"]
        println("[R=$(rank)] status=$(status) -> $(path)")
        push!(table_rows, payload)
    end

    println()
    println("| R  | EDdeg   | real_solns | best_frob | wall_time | status      |")
    println("|----|---------|------------|-----------|-----------|-------------|")
    for row in table_rows
        println(
            "| ", lpad(string(row["rank"]), 2),
            " | ", rpad(format_cell(row["ed_degree"]), 7),
            " | ", rpad(format_cell(row["real_solutions"]), 10),
            " | ", rpad(format_cell(row["best_frobenius"]), 9),
            " | ", rpad(format_cell(row["wall_time_seconds"]), 9),
            " | ", rpad(string(row["status"]), 11),
            " |"
        )
    end
end

main()