include("orbit_data.jl")
include("setup_system.jl")

using HomotopyContinuation

println("building R13 system")
problem = Main.SetupSystem.build_ed_problem(13)
println("variables=", length(problem.variables))
println("parameters=", length(problem.parameters))
println("equations=", length(problem.variables))

println("finding start pair")
start_pair = HomotopyContinuation.find_start_pair(problem.system; max_tries = 2_000)
println(start_pair === nothing ? "no start pair" : "start pair found")