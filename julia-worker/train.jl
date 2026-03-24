#!/usr/bin/env julia
# Standalone training script for 2048 RL agent
# Usage: julia --project=. train.jl [N] [episodes] [alpha]
#
# Examples:
#   julia --project=. train.jl              # N=4, 10000 episodes
#   julia --project=. train.jl 6 50000      # N=6, 50000 episodes
#   julia --project=. train.jl 6 100000 0.01  # N=6, 100K episodes, alpha=0.01

push!(LOAD_PATH, joinpath(@__DIR__, "src"))
using Game2048
using Game2048: QAgent, train!, save_weights, load_weights!, memory_mb
using Random

# Parse args
n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4
episodes = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10000
alpha = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.25

name = "agent_n$(n)"
weights_path = joinpath(@__DIR__, "weights", "$(name).bin")

# Create or load agent
agent = QAgent(name=name, n=n, alpha=alpha)

# Load existing weights if available
if isfile(weights_path)
    println("\nLoading existing weights...")
    load_weights!(agent, weights_path)
end

# Train
rng = MersenneTwister()
train!(agent, episodes; rng)

# Save
mkpath(dirname(weights_path))
save_weights(weights_path, agent)
