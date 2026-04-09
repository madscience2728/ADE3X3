include("orbit_data.jl")
include("setup_system.jl")
include("solve_rank.jl")

using JSON

payload = Main.SolveRank.solve_rank(13; debug = true)
println(JSON.json(payload))