include("orbit_data.jl")
include("setup_system.jl")
include("solve_rank.jl")

using JSON

if isempty(ARGS)
    error("usage: julia run_rank.jl <rank>")
end

rank = parse(Int, ARGS[1])
payload = Main.SolveRank.solve_rank(rank; debug = true)
path = Main.SolveRank.write_result(payload)
println(JSON.json(payload))
println(path)