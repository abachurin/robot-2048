#!/usr/bin/env julia
"""
Unclamped benchmark: frontier decay × clamp/no-clamp
4 configs, 1 run each, 500K episodes, N=6, alpha=0.15.
"""

push!(LOAD_PATH, joinpath(@__DIR__, "src"))
using Game2048
using Game2048: QAgent, episode!, memory_mb, max_cell, transfer_overhead_mb,
                set_n6_cutoff!, FRONTIER, print_board, Board
using Random
using Printf
using JSON3
using Dates

const EPISODES = 500_000
const CHECKPOINT = 100
const N_CHECKPOINTS = EPISODES ÷ CHECKPOINT

struct Config
    label::String
    cutoff::Int
    decay::Float64
    transfer::Bool
end

function run_config(cfg::Config, log_io::IO)
    set_n6_cutoff!(cfg.cutoff)
    agent = QAgent(name=cfg.label, n=6, alpha=0.15, decay=cfg.decay,
                   alpha_mode=FRONTIER, transfer=cfg.transfer, verbose=false)

    rng = MersenneTwister(42)
    curve = zeros(Float64, N_CHECKPOINTS)
    ma100 = Float64[]
    reached = zeros(Int, 7)
    best_score = Int32(0)
    best_board = Board(0)
    best_moves = Int32(0)

    t0 = time()
    for ep in 1:EPISODES
        game = episode!(agent, rng)
        push!(ma100, Float64(game.score))
        mt = max_cell(game.board)
        if mt >= 10
            reached[min(mt - 9, 7)] += 1
        end
        if game.score > best_score
            best_score = game.score
            best_board = game.board
            best_moves = game.num_moves
        end

        if ep % CHECKPOINT == 0
            curve[ep ÷ CHECKPOINT] = sum(ma100) / length(ma100)
            empty!(ma100)
        end

        if ep % 10000 == 0
            avg = sum(curve[max(1, ep÷CHECKPOINT-9):ep÷CHECKPOINT]) / min(10, ep÷CHECKPOINT)
            pct4 = sum(reached[3:7]) / ep * 100
            pct8 = sum(reached[4:7]) / ep * 100
            elapsed = time() - t0
            eta = elapsed / ep * (EPISODES - ep)
            msg = @sprintf("  %dk/%dk  avg=%.0f  4096=%.1f%%  8192=%.1f%%  ETA %.0fs",
                           ep÷1000, EPISODES÷1000, avg, pct4, pct8, eta)
            println(log_io, msg)
            flush(log_io)
            @printf("\r  %s: %s     ", cfg.label, msg)
            flush(stdout)
        end
    end
    elapsed = time() - t0
    println()

    final_avg = sum(curve[end-9:end]) / 10

    # Build milestones string
    milestones = String[]
    for (i, name) in enumerate(["1024", "2048", "4096", "8192", "16384", "32768", "65536"])
        pct = sum(reached[i:7]) / EPISODES * 100
        if pct > 0.01
            push!(milestones, @sprintf("%s=%.1f%%", name, pct))
        end
    end

    # Print summary + best game
    summary = @sprintf("  avg=%.0f  %s  %.0f ep/s  %.0f MB",
                       final_avg, join(milestones, "  "), EPISODES/elapsed,
                       memory_mb(agent.weights))
    println(summary)
    println(log_io, "  DONE: $summary")
    println(log_io, "  Best game: score=$(best_score), moves=$(best_moves), reached $(1 << max_cell(best_board))")
    best_io = IOBuffer()
    print_board(best_io, best_board)
    for line in split(String(take!(best_io)), '\n')
        !isempty(line) && println(log_io, "    $line")
    end
    println(log_io)
    flush(log_io)

    (curve=curve, final_avg=final_avg, elapsed=elapsed,
     pct_2048=sum(reached[2:7])/EPISODES*100,
     pct_4096=sum(reached[3:7])/EPISODES*100,
     pct_8192=sum(reached[4:7])/EPISODES*100,
     pct_16384=sum(reached[5:7])/EPISODES*100,
     memory_mb=memory_mb(agent.weights),
     eps_per_sec=EPISODES/elapsed,
     best_score=best_score)
end

function main()
    configs = [
        Config("d0.50",       15, 0.50, false),
        Config("d0.50+T",     15, 0.50, true),
        Config("d0.75",       15, 0.75, false),
        Config("d0.75+T",     15, 0.75, true),
    ]

    xs = collect(CHECKPOINT:CHECKPOINT:EPISODES)
    log_path = joinpath(@__DIR__, "logs", "benchmark_transfer.log")
    results_path = joinpath(@__DIR__, "logs", "benchmark_transfer.md")
    curves_path = joinpath(@__DIR__, "logs", "curves_transfer.json")
    mkpath(dirname(log_path))

    all_curves = Dict{String, Vector{Float64}}()
    all_results = Dict{String, NamedTuple}()

    open(log_path, "w") do log
        println(log, "Unclamped Benchmark — N=6, frontier alpha=0.15, $(EPISODES÷1000)K eps")
        println(log, "Started: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
        println(log, "")

        for cfg in configs
            println("\n$(cfg.label) (cutoff=$(cfg.cutoff), decay=$(cfg.decay)):")
            println(log, "$(cfg.label) (cutoff=$(cfg.cutoff), decay=$(cfg.decay)):")

            result = run_config(cfg, log)
            all_curves[cfg.label] = result.curve
            all_results[cfg.label] = result
        end

        println(log, "Finished: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    end

    # Save JSON
    json_data = Dict{String, Any}("episodes" => collect(xs), "curves" => all_curves)
    open(curves_path, "w") do f; JSON3.write(f, json_data); end

    # Write markdown
    open(results_path, "w") do io
        println(io, "# Unclamped Benchmark — N=6, frontier alpha=0.15, $(EPISODES÷1000)K episodes")
        println(io, "")
        println(io, "| Config | Transfer | Decay | Memory | Avg Score | 4096% | 8192% | Best | ep/s |")
        println(io, "|--------|----------|-------|--------|-----------|-------|-------|------|------|")
        for cfg in configs
            r = all_results[cfg.label]
            pct8 = r.pct_8192 > 0.05 ? @sprintf("%.1f%%", r.pct_8192) : "—"
            @printf(io, "| %s | %s | %.2f | %.0f MB | %.0f | %.1f%% | %s | %d | %.0f |\n",
                    cfg.label, cfg.transfer ? "yes" : "no", cfg.decay, r.memory_mb,
                    r.final_avg, r.pct_4096, pct8, r.best_score, r.eps_per_sec)
        end
        println(io, "")
    end

    # Charts
    try
        run(`python3 $(joinpath(@__DIR__, "plot_transfer.py")) $curves_path $(joinpath(@__DIR__, "logs")) $results_path`)
    catch e
        println("Warning: chart generation failed: $e")
    end

    println("\nResults: $results_path")
end

main()
