#!/usr/bin/env julia
"""
Float precision benchmark: Float32 vs Float64
For each (N, FloatType): 24 runs of 20K episodes.
Trim 2 best + 2 worst by final avg score → average remaining 20 learning curves.
Output: charts + table + log file with progress.
"""

push!(LOAD_PATH, joinpath(@__DIR__, "src"))
using Game2048
using Game2048: QAgent, episode!, memory_mb, max_cell
using Random
using Printf
using JSON3

const EPISODES = 20_000
const CHECKPOINT = 100   # record avg every 100 episodes
const N_RUNS = 100
const TRIM = 5
const N_CHECKPOINTS = EPISODES ÷ CHECKPOINT

# Run one 20K-episode training, return learning curve + final stats
function run_single(n::Int, float_type::Type{T}, seed::Int) where T
    agent = QAgent(name="b", n=n, float_type=T, verbose=false)
    rng = MersenneTwister(seed)

    curve = zeros(Float64, N_CHECKPOINTS)  # avg score at each 100-ep checkpoint
    ma100 = Float64[]
    reached_2048 = 0
    reached_4096 = 0
    total_games = 0

    for ep in 1:EPISODES
        game = episode!(agent, rng)
        push!(ma100, Float64(game.score))
        mt = max_cell(game.board)
        if mt >= 11; reached_2048 += 1; end
        if mt >= 12; reached_4096 += 1; end
        total_games += 1

        if ep % CHECKPOINT == 0
            idx = ep ÷ CHECKPOINT
            curve[idx] = sum(ma100) / length(ma100)
            empty!(ma100)
        end

        # Decay every 1000 episodes
        if ep % 1000 == 0 && agent.alpha > agent.min_alpha
            agent.alpha = max(agent.alpha * agent.decay, agent.min_alpha)
        end
    end

    final_avg = sum(curve[end-9:end]) / 10  # avg of last 1000 eps
    pct_2048 = reached_2048 / total_games * 100
    pct_4096 = reached_4096 / total_games * 100
    (curve=curve, final_avg=final_avg, pct_2048=pct_2048, pct_4096=pct_4096)
end

function trimmed_indices(finals::Vector{Float64}; trim::Int=TRIM)
    order = sortperm(finals)
    order[trim+1:end-trim]
end

function benchmark_all()
    float_types = [Float32, Float64]
    n_values = [2, 3, 4, 5, 6]
    xs = collect(CHECKPOINT:CHECKPOINT:EPISODES)

    log_path = joinpath(@__DIR__, "logs", "benchmark_float_progress.log")
    results_path = joinpath(@__DIR__, "logs", "benchmark_float_types.md")
    mkpath(dirname(log_path))

    # Storage: (T, n) → averaged curve + stats + all raw finals (for histograms)
    avg_curves = Dict{Tuple{Type, Int}, Vector{Float64}}()
    all_finals_data = Dict{Tuple{Type, Int}, Vector{Float64}}()
    stats = Dict{Tuple{Type, Int}, NamedTuple{(:avg_score, :pct_2048, :pct_4096, :time_s, :mem_mb),
                                               Tuple{Float64, Float64, Float64, Float64, Float64}}}()

    open(log_path, "w") do log
        total_start = time()
        header = "Float Precision Benchmark — $(EPISODES) episodes × $N_RUNS runs, trim $TRIM best + $TRIM worst"
        println(header)
        println(log, header)
        println(log, "Started: $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))")
        println(log, "")

        for T in float_types
            for n in n_values
                mb = memory_mb(QAgent(name="x", n=n, float_type=T, verbose=false).weights)
                label = "N=$n $T ($(round(mb, digits=1)) MB)"
                msg = "$label: running $N_RUNS × $(EPISODES) episodes..."
                println(msg)
                println(log, msg)
                flush(log)

                all_curves = Vector{Vector{Float64}}()
                all_finals = Float64[]
                all_2048 = Float64[]
                all_4096 = Float64[]
                t0 = time()

                for run in 1:N_RUNS
                    seed = 7919 * n + 997 * run + (T == Float32 ? 0 : 500)
                    result = run_single(n, T, seed)
                    push!(all_curves, result.curve)
                    push!(all_finals, result.final_avg)
                    push!(all_2048, result.pct_2048)
                    push!(all_4096, result.pct_4096)

                    elapsed = time() - t0
                    eta = elapsed / run * (N_RUNS - run)
                    progress = "  run $run/$N_RUNS done ($(round(elapsed, digits=1))s elapsed, ETA $(round(Int, eta))s)"
                    println(progress)
                    println(log, progress)
                    flush(log)
                end

                elapsed = time() - t0

                # Trim 2 best + 2 worst by final_avg
                keep = trimmed_indices(all_finals)
                kept_curves = [all_curves[i] for i in keep]
                avg_curve = zeros(Float64, N_CHECKPOINTS)
                for c in kept_curves
                    avg_curve .+= c
                end
                avg_curve ./= length(kept_curves)

                avg_curves[(T, n)] = avg_curve
                all_finals_data[(T, n)] = copy(all_finals)
                stats[(T, n)] = (
                    avg_score = sum(all_finals[keep]) / length(keep),
                    pct_2048 = sum(all_2048[keep]) / length(keep),
                    pct_4096 = sum(all_4096[keep]) / length(keep),
                    time_s = elapsed / N_RUNS,
                    mem_mb = mb,
                )

                summary = "  $label done in $(round(elapsed, digits=1))s — avg=$(round(Int, stats[(T,n)].avg_score)), 2048=$(round(stats[(T,n)].pct_2048, digits=1))%, 4096=$(round(stats[(T,n)].pct_4096, digits=1))%"
                println(summary)
                println(log, summary)
                println(log, "")
                flush(log)
            end
        end

        total_time = time() - total_start
        footer = "\nTotal benchmark time: $(round(total_time / 60, digits=1)) minutes"
        println(footer)
        println(log, footer)
    end

    # === Write markdown results ===
    open(results_path, "w") do io
        println(io, "# Float Precision Benchmark — Float32 vs Float64")
        println(io, "")
        println(io, "$(EPISODES) episodes × $N_RUNS runs per cell, trimmed mean (drop $TRIM best + $TRIM worst = $(N_RUNS - 2*TRIM) averaged)")
        println(io, "Training params: alpha=0.25, decay=0.9 every 1000 eps, min_alpha=0.01")
        println(io, "")

        function fmt_val(field::Symbol, val::Float64)
            if field == :avg_score
                @sprintf("%7.0f", val)
            elseif field in (:pct_2048, :pct_4096)
                @sprintf("%5.1f%%", val)
            elseif field == :time_s
                @sprintf("%5.1fs", val)
            else
                @sprintf("%7.1f", val)
            end
        end

        for (title, field) in [
            ("Average Score (last 1K episodes)", :avg_score),
            ("2048 Reached %", :pct_2048),
            ("4096 Reached %", :pct_4096),
            ("Time per 20K run (seconds)", :time_s),
            ("Weight Memory (MB)", :mem_mb),
        ]
            println(io, "## $title")
            println(io, "")
            println(io, "| N | Float32 | Float64 |")
            println(io, "|---|---------|---------|")
            for n in n_values
                f32 = getfield(stats[(Float32, n)], field)
                f64 = getfield(stats[(Float64, n)], field)
                println(io, "| $n | $(fmt_val(field, f32)) | $(fmt_val(field, f64)) |")
            end
            println(io, "")
        end
    end
    # === Save curve data + raw finals as JSON ===
    curves_path = joinpath(@__DIR__, "logs", "curves_data.json")
    curves_dict = Dict{String, Any}(
        "episodes" => collect(xs),
        "n_runs" => N_RUNS,
        "trim" => TRIM,
    )
    curve_data = Dict{String, Vector{Float64}}()
    finals_data = Dict{String, Vector{Float64}}()
    for T in float_types
        for n in n_values
            key = "N=$(n)_$(nameof(T))"
            curve_data[key] = avg_curves[(T, n)]
            finals_data[key] = all_finals_data[(T, n)]
        end
    end
    curves_dict["curves"] = curve_data
    curves_dict["finals"] = finals_data
    open(curves_path, "w") do f
        JSON3.write(f, curves_dict)
    end
    println("Curve data: $curves_path")

    # === Generate PNG charts via Python/matplotlib ===
    plot_script = joinpath(@__DIR__, "plot_curves.py")
    logs_dir = joinpath(@__DIR__, "logs")
    try
        run(`python3 $plot_script $curves_path $logs_dir`)
        println("PNG charts saved to $logs_dir/")

        # Add charts to markdown
        open(results_path, "a") do io
            println(io, "## Learning Curves")
            println(io, "")
            for n in n_values
                println(io, "### N=$n")
                println(io, "![N=$n](curve_n$n.png)")
                println(io, "")
            end
            println(io, "### All N values (Float32)")
            println(io, "![All N](curve_all_n.png)")
            println(io, "")
            println(io, "## Score Distributions (log scale)")
            println(io, "")
            for n in n_values
                println(io, "### N=$n")
                println(io, "![N=$n hist](hist_n$n.png)")
                println(io, "")
            end
            println(io, "### All N values")
            println(io, "![All N hist](hist_all_n.png)")
        end
    catch e
        println("Warning: PNG generation failed: $e")
    end

    println("\nResults table: $results_path")
    println("Log: $(joinpath(@__DIR__, "logs", "benchmark_float_progress.log"))")
end

using Dates
benchmark_all()
