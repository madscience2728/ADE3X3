module OrbitData

export RankSpec, ALL_RANKS, rank_spec, result_path

struct RankSpec
    rank::Int
    orbit_config::Vector{Int}
    eff_params::Int
    generic_h_rank::Union{Int, Nothing}
    target_h_rank::Union{Int, Nothing}
    trivial_placeholder::Bool
    note::String
end

const ALL_RANKS = [13, 19, 20, 21, 22, 23, 27]

const RANK_SPECS = Dict(
    13 => RankSpec(13, [1, 12], 15, 3, 4, false,
        "One-orbit symmetric chart is fully specified. R13 is the first solve target."),
    19 => RankSpec(19, [1, 6, 12], 24, 7, 10, false,
        "One-orbit symmetric chart is fully specified. Higher-rank solve not attempted yet in this scaffold."),
    20 => RankSpec(20, [8, 12], 18, 6, 11, false,
        "One-orbit symmetric chart is fully specified. Higher-rank solve not attempted yet in this scaffold."),
    21 => RankSpec(21, [1, 8, 12], 21, 7, 12, false,
        "One-orbit symmetric chart is fully specified. Higher-rank solve not attempted yet in this scaffold."),
    22 => RankSpec(22, [1, 6, 8, 12], 30, nothing, nothing, false,
        "One-orbit symmetric chart is fully specified. Higher-rank solve not attempted yet in this scaffold."),
    23 => RankSpec(23, Int[], 0, nothing, nothing, true,
        "Exact decomposition exists (Laderman). Emitted as requested placeholder."),
    27 => RankSpec(27, Int[], 0, nothing, nothing, true,
        "Standard algorithm. Emitted as requested placeholder."),
)

function rank_spec(rank::Int)
    haskey(RANK_SPECS, rank) || error("Unsupported rank: $(rank)")
    return RANK_SPECS[rank]
end

result_path(rank::Int) = joinpath(@__DIR__, "results", "ed_R$(rank).json")

end