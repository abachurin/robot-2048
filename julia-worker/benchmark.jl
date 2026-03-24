#!/usr/bin/env julia
"""
Standalone Julia RL trainer benchmark.
Usage: julia --project=. benchmark.jl [N] [episodes]
Writes log to benchmark_nN.log
"""

push!(LOAD_PATH, joinpath(@__DIR__, "src"))
using Game2048
using Game2048: QAgent, episode!, evaluate, new_game, game_over, max_cell,
                empty_count, memory_mb, Board, matrix_to_board, print_board,
                d4_symmetries, make_move
using Random

function run_benchmark(n::Int, episodes::Int; alpha::Float64=0.25)
    log_path = joinpath(@__DIR__, "benchmark_n$(n).log")
    log_lines = String[]

    function log(msg)
        println(msg)
        push!(log_lines, msg)
    end

    log("Julia benchmark: N=$n, alpha=$alpha, episodes=$episodes")
    t0 = time()
    agent = QAgent(name="bench_n$n", n=n, alpha=alpha)
    log("Weights: $(length(agent.weights.data)) entries ($(round(memory_mb(agent.weights), digits=1)) MB)")
    log("Init time: $(round(time() - t0, digits=2))s\n")

    start = time()
    start_1000 = start
    ma100 = Float64[]
    av1000 = Float64[]
    reached = zeros(Int, 7)
    best_of_1000_score = Int32(0)
    best_of_1000_str = ""
    rng = MersenneTwister()

    for ep in 1:episodes
        game = episode!(agent, rng)
        push!(ma100, game.score)
        push!(av1000, game.score)
        mt = max_cell(game.board)

        game_str = let io = IOBuffer()
            print_board(io, game.board)
            s = String(take!(io))
            s * " score = $(game.score), moves = $(game.num_moves), reached $(1 << mt)\n"
        end

        if game.score > best_of_1000_score
            best_of_1000_score = game.score
            best_of_1000_str = game_str
            if game.score > agent.best_score
                agent.best_score = game.score
                log("\nNew best game at episode $(ep)!\n$game_str")
            end
        end

        if mt >= 10
            reached[min(mt - 9, 7)] += 1
        end
        agent.max_tile = max(agent.max_tile, mt)

        if ep % 100 == 0
            average = round(Int, sum(ma100) / length(ma100))
            log("episode $ep, last 100 average = $average")
            empty!(ma100)
        end

        if ep % 1000 == 0
            average = round(Int, sum(av1000) / length(av1000))
            elapsed_1000 = time() - start_1000
            log("\n=== Episode $ep ===")
            log("$(round(elapsed_1000, digits=1))s for last $(length(av1000)) episodes")
            log("average score = $average")
            for j in 1:7
                r = round(sum(reached[j:7]) / length(av1000) * 100, digits=2)
                r > 0 && log("$(1 << (j + 9)) reached in $(r)%")
            end
            log("best game of last 1000:\n$best_of_1000_str")
            log("best game ever: score=$(agent.best_score)")
            empty!(av1000)
            fill!(reached, 0)
            best_of_1000_score = Int32(0)
            best_of_1000_str = ""
            start_1000 = time()
        end
    end

    total = time() - start
    log("\nTotal time: $(round(total, digits=1))s ($(round(Int, episodes / total)) episodes/sec)")

    open(log_path, "w") do f
        write(f, join(log_lines, "\n"))
    end
    println("\nLog written to $log_path")
end

n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4
episodes = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1000
run_benchmark(n, episodes)
