#!/usr/bin/env julia
"""
Compare training strategies on N=6, 200K episodes:
  1. global_decay: alpha=0.25, decay=0.9, step=1000 (original)
  2. constant: alpha=0.1, no decay
  3. frontier: alpha=0.25, decay=0.5 per frontier gap level

Also compare N=6 cutoff values: 13 (original), 14, 15 (no clamp)

Saves results to logs/benchmark_strategies.md with PNG charts.
"""

push!(LOAD_PATH, joinpath(@__DIR__, "src"))
using Game2048
using Game2048: QAgent, episode!, memory_mb, max_cell, set_n6_cutoff!, N6_CUTOFF,
                GLOBAL_DECAY, CONSTANT, FRONTIER, AlphaMode
using Random
using Printf
using JSON3
using Dates

const EPISODES = 200_000
const CHECKPOINT = 100
const N_CHECKPOINTS = EPISODES ÷ CHECKPOINT
const N_RUNS = 5   # average over 5 runs for each config

struct Config
    label::String
    n::Int
    alpha::Float64
    decay::Float64
    step::Int
    min_alpha::Float64
    alpha_mode::AlphaMode
    cutoff::Int  # N=6 cutoff (13, 14, 15)
end

function run_config(cfg::Config, seed::Int)
    set_n6_cutoff!(cfg.cutoff)
    agent = QAgent(name=cfg.label, n=cfg.n, alpha=cfg.alpha, decay=cfg.decay,
                   step=cfg.step, min_alpha=cfg.min_alpha, alpha_mode=cfg.alpha_mode,
                   verbose=false)
    rng = MersenneTwister(seed)

    curve = zeros(Float64, N_CHECKPOINTS)
    ma100 = Float64[]
    reached_2048 = 0
    reached_4096 = 0
    reached_8192 = 0

    for ep in 1:EPISODES
        game = episode!(agent, rng)
        push!(ma100, Float64(game.score))
        mt = max_cell(game.board)
        mt >= 11 && (reached_2048 += 1)
        mt >= 12 && (reached_4096 += 1)
        mt >= 13 && (reached_8192 += 1)

        if ep % CHECKPOINT == 0
            curve[ep ÷ CHECKPOINT] = sum(ma100) / length(ma100)
            empty!(ma100)
        end

        # Global decay
        if cfg.alpha_mode == GLOBAL_DECAY && ep % cfg.step == 0 && agent.alpha > cfg.min_alpha
            agent.alpha = max(agent.alpha * cfg.decay, cfg.min_alpha)
        end
    end

    final_avg = sum(curve[end-9:end]) / 10
    (curve=curve, final_avg=final_avg,
     pct_2048=reached_2048/EPISODES*100,
     pct_4096=reached_4096/EPISODES*100,
     pct_8192=reached_8192/EPISODES*100,
     memory_mb=memory_mb(agent.weights))
end

function main()
    configs = [
        # Alpha mode comparison (cutoff=13)
        Config("global_decay", 6, 0.25, 0.9, 1000, 0.01, GLOBAL_DECAY, 13),
        Config("constant_0.10", 6, 0.10, 0.0, 1000, 0.10, CONSTANT, 13),
        Config("constant_0.15", 6, 0.15, 0.0, 1000, 0.15, CONSTANT, 13),
        Config("frontier_d0.5", 6, 0.25, 0.5, 1000, 0.01, FRONTIER, 13),
        Config("frontier_d0.7", 6, 0.25, 0.7, 1000, 0.01, FRONTIER, 13),
        # Cutoff comparison (global_decay)
        Config("cutoff_14", 6, 0.25, 0.9, 1000, 0.01, GLOBAL_DECAY, 14),
        Config("cutoff_15", 6, 0.25, 0.9, 1000, 0.01, GLOBAL_DECAY, 15),
    ]

    xs = collect(CHECKPOINT:CHECKPOINT:EPISODES)
    log_path = joinpath(@__DIR__, "logs", "benchmark_strategies_progress.log")
    results_path = joinpath(@__DIR__, "logs", "benchmark_strategies.md")
    curves_path = joinpath(@__DIR__, "logs", "curves_strategies.json")
    mkpath(dirname(log_path))

    avg_curves = Dict{String, Vector{Float64}}()
    results = Dict{String, NamedTuple}()

    open(log_path, "w") do log
        println("Strategy Benchmark — N=6, $(EPISODES) episodes, $N_RUNS runs each")
        println(log, "Strategy Benchmark — N=6, $(EPISODES) episodes, $N_RUNS runs each")
        println(log, "Started: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
        println(log, "")

        for cfg in configs
            println("\n$(cfg.label) (cutoff=$(cfg.cutoff)):")
            println(log, "$(cfg.label) (cutoff=$(cfg.cutoff)):")
            flush(log)

            all_curves = Vector{Vector{Float64}}()
            all_finals = Float64[]
            all_2048 = Float64[]
            all_4096 = Float64[]
            all_8192 = Float64[]
            mem = 0.0
            t0 = time()

            for run in 1:N_RUNS
                result = run_config(cfg, 42 + run * 1000)
                push!(all_curves, result.curve)
                push!(all_finals, result.final_avg)
                push!(all_2048, result.pct_2048)
                push!(all_4096, result.pct_4096)
                push!(all_8192, result.pct_8192)
                mem = result.memory_mb
                elapsed = time() - t0
                eta = elapsed / run * (N_RUNS - run)
                msg = "  run $run/$N_RUNS: avg=$(round(Int, result.final_avg)), 4096=$(round(result.pct_4096, digits=1))% ($(round(Int, eta))s remaining)"
                println(msg)
                println(log, msg)
                flush(log)
            end

            # Average curves
            avg_curve = zeros(Float64, N_CHECKPOINTS)
            for c in all_curves
                avg_curve .+= c
            end
            avg_curve ./= N_RUNS

            avg_curves[cfg.label] = avg_curve
            results[cfg.label] = (
                avg_score = sum(all_finals) / N_RUNS,
                pct_2048 = sum(all_2048) / N_RUNS,
                pct_4096 = sum(all_4096) / N_RUNS,
                pct_8192 = sum(all_8192) / N_RUNS,
                time_s = (time() - t0) / N_RUNS,
                memory_mb = mem,
                cutoff = cfg.cutoff,
            )

            elapsed = time() - t0
            summary = "  $(cfg.label) done in $(round(elapsed, digits=0))s — avg=$(round(Int, results[cfg.label].avg_score)), 4096=$(round(results[cfg.label].pct_4096, digits=1))%, mem=$(round(mem, digits=0))MB"
            println(summary)
            println(log, summary)
            println(log, "")
            flush(log)
        end
    end

    # === Save JSON ===
    json_data = Dict{String, Any}(
        "episodes" => collect(xs),
        "curves" => avg_curves,
    )
    open(curves_path, "w") do f
        JSON3.write(f, json_data)
    end

    # === Write markdown ===
    open(results_path, "w") do io
        println(io, "# Training Strategy Comparison — N=6, $(EPISODES) episodes")
        println(io, "")
        println(io, "Averaged over $N_RUNS runs per config")
        println(io, "")
        println(io, "## Results")
        println(io, "")
        println(io, "| Strategy | Cutoff | Memory | Avg Score | 2048% | 4096% | 8192% | sec/run |")
        println(io, "|----------|--------|--------|-----------|-------|-------|-------|---------|")
        for cfg in configs
            r = results[cfg.label]
            @printf(io, "| %s | %d | %4.0f MB | %6.0f | %5.1f%% | %5.1f%% | %5.1f%% | %5.0fs |\n",
                    cfg.label, r.cutoff, r.memory_mb, r.avg_score, r.pct_2048, r.pct_4096, r.pct_8192, r.time_s)
        end
        println(io, "")
    end

    # === Generate charts ===
    plot_script = joinpath(@__DIR__, "plot_strategies.py")
    try
        run(`python3 $plot_script $curves_path $(joinpath(@__DIR__, "logs")) $results_path`)
    catch e
        println("Warning: plot generation failed: $e")
    end

    println("\nResults: $results_path")
    println("Log: $log_path")
end

main()
