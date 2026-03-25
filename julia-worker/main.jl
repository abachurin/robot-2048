#!/usr/bin/env julia
# Julia worker — polls the API for pending jobs and executes them
#
# Usage: julia --project=. main.jl [--host localhost] [--port 5000]

push!(LOAD_PATH, joinpath(@__DIR__, "src"))

using Game2048
using Game2048: QAgent, episode!, new_game, game_over, make_move, find_best_move,
                add_random_tile!, find_empty_cells, set_cell, get_cell, max_cell,
                evaluate, save_weights, load_weights!, weight_group_sizes, decay_alpha!,
                d4_symmetries, update_weights!, board_to_matrix, Board, GameState,
                trial_run!, GLOBAL_DECAY
using Random

include("src/worker_client.jl")
include("src/job_runner.jl")

const POLL_INTERVAL = 3.0  # seconds

function parse_args()
    host = "localhost"
    port = 8000
    args = ARGS
    i = 1
    while i <= length(args)
        if args[i] == "--host" && i < length(args)
            host = args[i+1]
            i += 2
        elseif args[i] == "--port" && i < length(args)
            port = parse(Int, args[i+1])
            i += 2
        else
            i += 1
        end
    end
    host, port
end

function main()
    host, port = parse_args()
    client = WorkerClient(; host, port)

    # Weights cache directory
    weights_dir = joinpath(@__DIR__, "weights")
    mkpath(weights_dir)

    println("Julia worker starting (connecting to $host:$port)")

    # Wait for API to be ready
    while true
        try
            get_jobs(client)
            break
        catch e
            println("Waiting for API... ($(typeof(e).name.name))")
            sleep(2)
        end
    end
    println("API connected. Cleaning up stale watch jobs...")

    cleanup(client)

    # Track running jobs: description => Task
    running = Dict{String, Task}()

    println("Worker ready. Polling every $(POLL_INTERVAL)s...\n")

    while true
        try
            jobs = get_jobs(client)

            # Get current state
            active_descriptions = Set{String}()
            pending = String[]
            if jobs !== nothing
                for job in jobs
                    desc = String(job.description)
                    push!(active_descriptions, desc)
                    if Int(job.status) == JOB_STATUS_PENDING
                        push!(pending, desc)
                    end
                end
            end

            # Clean up finished tasks
            for (desc, task) in collect(running)
                if istaskdone(task)
                    if istaskfailed(task)
                        try
                            fetch(task)
                        catch e
                            println(stderr, "Task '$desc' failed: $e")
                        end
                    end
                    delete!(running, desc)
                    println("Job finished: $desc")
                end
                # If job was deleted from DB but task still running, it will
                # notice on next status check and stop
            end

            # Launch pending jobs
            for desc in pending
                if !haskey(running, desc)
                    job = launch_job(client, desc)
                    if job === nothing
                        println("Job '$desc' disappeared before launch")
                        continue
                    end
                    println("Launching: $desc")
                    task = @task run_job(client, job, weights_dir)
                    running[desc] = task
                    schedule(task)
                end
            end

        catch e
            if e isa HTTP.Exceptions.ConnectError || e isa Base.IOError
                println(stderr, "API connection lost, retrying...")
            else
                println(stderr, "Worker error: $(sprint(showerror, e))")
            end
        end

        sleep(POLL_INTERVAL)
    end
end

main()
