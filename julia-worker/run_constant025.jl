#!/usr/bin/env julia
"""
Run constant alpha=0.25 to compare fairly with frontier d=0.5 and d=0.7 (same base alpha).
Appends results to the strategies log and JSON.
"""

push!(LOAD_PATH, joinpath(@__DIR__, "src"))
using Game2048
using Game2048: QAgent, episode!, memory_mb, max_cell, CONSTANT
using Random
using Printf
using JSON3
using Dates

const EPISODES = 200_000
const CHECKPOINT = 100
const N_CHECKPOINTS = EPISODES ÷ CHECKPOINT
const N_RUNS = 5

function run_single(seed::Int)
    agent = QAgent(name="c025", n=6, alpha=0.25, alpha_mode=CONSTANT, verbose=false)
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
    end

    final_avg = sum(curve[end-9:end]) / 10
    (curve=curve, final_avg=final_avg,
     pct_2048=reached_2048/EPISODES*100,
     pct_4096=reached_4096/EPISODES*100,
     pct_8192=reached_8192/EPISODES*100)
end

function main()
    log_path = joinpath(@__DIR__, "logs", "benchmark_strategies_progress.log")
    curves_path = joinpath(@__DIR__, "logs", "curves_strategies.json")

    println("constant_0.25: running $N_RUNS × $EPISODES episodes...")
    open(log_path, "a") do log
        println(log, "\nconstant_0.25 (cutoff=13):")
    end

    all_curves = Vector{Vector{Float64}}()
    all_finals = Float64[]
    all_2048 = Float64[]
    all_4096 = Float64[]
    all_8192 = Float64[]
    t0 = time()

    for run in 1:N_RUNS
        result = run_single(42 + run * 1000)
        push!(all_curves, result.curve)
        push!(all_finals, result.final_avg)
        push!(all_2048, result.pct_2048)
        push!(all_4096, result.pct_4096)
        push!(all_8192, result.pct_8192)
        elapsed = time() - t0
        eta = elapsed / run * (N_RUNS - run)
        msg = "  run $run/$N_RUNS: avg=$(round(Int, result.final_avg)), 4096=$(round(result.pct_4096, digits=1))% ($(round(Int, eta))s remaining)"
        println(msg)
        open(log_path, "a") do log; println(log, msg); end
    end

    # Average curves
    avg_curve = zeros(Float64, N_CHECKPOINTS)
    for c in all_curves; avg_curve .+= c; end
    avg_curve ./= N_RUNS

    elapsed = time() - t0
    summary = "  constant_0.25 done in $(round(elapsed, digits=0))s — avg=$(round(Int, sum(all_finals)/N_RUNS)), 4096=$(round(sum(all_4096)/N_RUNS, digits=1))%, 8192=$(round(sum(all_8192)/N_RUNS, digits=1))%"
    println(summary)
    open(log_path, "a") do log; println(log, summary); println(log, ""); end

    # Append curve to existing JSON
    if isfile(curves_path)
        data = JSON3.read(read(curves_path, String))
        curves = Dict{String, Any}(String(k) => v for (k, v) in pairs(data.curves))
        curves["constant_0.25"] = avg_curve
        new_data = Dict("episodes" => data.episodes, "curves" => curves)
        open(curves_path, "w") do f; JSON3.write(f, new_data); end
        println("Updated $curves_path")
    end
end

main()
