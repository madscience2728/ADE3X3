include("orbit_data.jl")
include("setup_system.jl")
include("solve_rank.jl")

using JSON

payload = Main.SolveRank.solve_rank(22; debug = true)
path = Main.SolveRank.write_result(payload)
println(JSON.json(payload))
println(path)