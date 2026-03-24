using Test

# Load the module
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Game2048

@testset "Game2048" begin
    include("test_board.jl")
    include("test_lookup.jl")
    include("test_symmetry.jl")
    include("test_features.jl")
    include("test_game.jl")
    include("test_agent.jl")
end
