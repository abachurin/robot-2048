# Job runner — executes train/test/watch jobs using the Game2048 engine
# and communicates progress back to the API via WorkerClient

const JOB_TYPE_TRAIN = 0
const JOB_TYPE_TEST = 1
const JOB_TYPE_WATCH = 2

const JOB_STATUS_PENDING = 0
const JOB_STATUS_RUN = 1
const JOB_STATUS_STOP = 2

# Convert board matrix (4x4 list-of-lists) to our Board representation
function matrix_to_board_from_json(matrix)::Board
    board = Board(0)
    for i in 0:3
        for j in 0:3
            val = Int(matrix[i+1][j+1])
            if val > 0
                # Convert tile value to exponent: 2=1, 4=2, 8=3, etc.
                exp = Int(round(log2(val)))
                board = set_cell(board, i, j, exp)
            end
        end
    end
    board
end

# Convert Board to 4x4 matrix of tile values (0, 2, 4, 8, ...)
function board_to_value_matrix(board::Board)
    matrix = zeros(Int, 4, 4)
    for i in 0:3
        for j in 0:3
            c = get_cell(board, i, j)
            matrix[i+1, j+1] = c == 0 ? 0 : 1 << c
        end
    end
    [matrix[i, :] for i in 1:4]  # return as list of rows
end

# ============================================================================
# Training job
# ============================================================================
function run_train_job(client::WorkerClient, job::Any, weights_dir::String)
    description = String(job.description)
    user = String(job.user)
    agent_name = String(job.name)
    episodes = Int(job.episodes)
    n = Int(job.N)
    alpha = Float64(job.alpha)
    decay = Float64(job.decay)
    step = Int(job.step)
    min_alpha = Float64(job.minAlpha)

    add_log(client, user, "Starting training: $agent_name, $episodes episodes")

    # Create agent
    agent = QAgent(; name=agent_name, n=n, alpha=alpha, decay=decay,
                   step=step, min_alpha=min_alpha, verbose=false)

    # Try to load existing weights
    weights_path = joinpath(weights_dir, "$agent_name.bin")
    found = download_weights(client, agent_name, weights_path)
    if found
        load_weights!(agent, weights_path)
        add_log(client, user, "Loaded existing weights (episode $(agent.last_episode))")
    end

    rng = MersenneTwister()
    start_time = time()
    first_episode = agent.last_episode
    last_episode = agent.last_episode + episodes
    checkpoint_interval = 1000
    log_interval = 100

    ma100 = Float64[]
    best_game = new_game(rng)

    while agent.last_episode < last_episode
        # Check for stop/kill
        status = get_job_status(client, description)
        if status === nothing || status == JOB_STATUS_STOP
            add_log(client, user, "Job stopped at episode $(agent.last_episode)")
            break
        end

        game = episode!(agent, rng)
        agent.last_episode += 1
        push!(ma100, game.score)

        if game.score > best_game.score
            best_game = game
        end
        if game.score > agent.best_score
            agent.best_score = game.score
        end
        mt = max_cell(game.board)
        agent.max_tile = max(agent.max_tile, mt)

        # History collection
        if agent.last_episode % 100 == 0
            avg = round(Int, sum(ma100) / length(ma100))
            if agent.last_episode % agent.collect_step == 0
                push!(agent.history, avg)
                if length(agent.history) >= agent.max_train_history
                    agent.history = agent.history[1:2:end]
                    agent.collect_step *= 2
                end
            end
            empty!(ma100)
        end

        # Log every log_interval episodes
        if agent.last_episode % log_interval == 0
            elapsed = time() - start_time
            eps_done = agent.last_episode - first_episode
            eps_remaining = last_episode - agent.last_episode
            eta = eps_remaining > 0 ? round(Int, elapsed / eps_done * eps_remaining) : 0

            add_log(client, user,
                "Episode $(agent.last_episode), best=$(agent.best_score), " *
                "reached=$(1 << agent.max_tile), alpha=$(round(agent.alpha, digits=6))")

            update_timing(client, description, round(Int, elapsed), eta)
        end

        # Checkpoint every checkpoint_interval
        if agent.last_episode % checkpoint_interval == 0
            save_weights(weights_path, agent)
            upload_weights(client, agent_name, weights_path)
            update_agent(client, agent_name, Dict(
                "bestScore" => agent.best_score,
                "maxTile" => agent.max_tile,
                "alpha" => agent.alpha,
                "lastTrainingEpisode" => agent.last_episode,
                "history" => agent.history,
                "collectStep" => agent.collect_step,
                "weightSignature" => weight_group_sizes(agent.n)
            ))
        end

        # Learning rate decay
        if agent.alpha_mode == GLOBAL_DECAY
            if agent.last_episode >= agent.next_decay && agent.alpha > agent.min_alpha
                decay_alpha!(agent)
                update_alpha(client, description, agent.alpha)
                add_log(client, user, "LR decayed to $(round(agent.alpha, digits=6))")
            end
        end
    end

    # Final save
    save_weights(weights_path, agent)
    upload_weights(client, agent_name, weights_path)
    update_agent(client, agent_name, Dict(
        "bestScore" => agent.best_score,
        "maxTile" => agent.max_tile,
        "alpha" => agent.alpha,
        "lastTrainingEpisode" => agent.last_episode,
        "history" => agent.history,
        "collectStep" => agent.collect_step,
        "weightSignature" => weight_group_sizes(agent.n)
    ))

    elapsed = time() - start_time
    eps_done = agent.last_episode - first_episode
    rate = eps_done > 0 ? round(eps_done / elapsed, digits=0) : 0
    add_log(client, user,
        "Training complete: $(eps_done) episodes in $(round(elapsed, digits=1))s " *
        "($(rate) eps/sec), best=$(agent.best_score)")
end

# ============================================================================
# Test job
# ============================================================================
function run_test_job(client::WorkerClient, job::Any, weights_dir::String)
    description = String(job.description)
    user = String(job.user)
    agent_name = String(job.name)
    episodes = Int(job.episodes)
    depth = Int(something(get(job, :depth, 0), 0))
    width = Int(something(get(job, :width, 1), 1))
    trigger = Int(something(get(job, :trigger_, 0), 0))

    add_log(client, user, "Starting test: $agent_name, $episodes games (d=$depth, w=$width, t=$trigger)")

    # Load agent info to get N
    agent_info = get_agent(client, agent_name)
    if agent_info === nothing
        add_log(client, user, "Agent $agent_name not found")
        return
    end
    n = Int(agent_info.N)

    # Handle special agents
    if agent_name == "Random Moves"
        evaluator = _ -> rand()
    elseif agent_name == "Best Score"
        evaluator = _ -> Float64(0)
    else
        agent = QAgent(; name=agent_name, n=n, verbose=false)
        weights_path = joinpath(weights_dir, "$agent_name.bin")
        found = download_weights(client, agent_name, weights_path)
        if found
            load_weights!(agent, weights_path)
        end
        evaluator = board -> evaluate(agent, board)
    end

    rng = MersenneTwister()
    start_time = time()
    scores = Int32[]
    tiles = Int[]
    best_game = new_game(rng)

    for i in 1:episodes
        status = get_job_status(client, description)
        if status === nothing || status == JOB_STATUS_STOP
            add_log(client, user, "Test stopped after $i games")
            break
        end

        game = new_game(rng)
        trial_run!(game, evaluator; depth, width, trigger, rng)

        mt = max_cell(game.board)
        push!(scores, game.score)
        push!(tiles, mt)
        if game.score > best_game.score
            best_game = game
        end

        elapsed = time() - start_time
        remaining = i < episodes ? round(Int, elapsed / i * (episodes - i)) : 0
        update_timing(client, description, round(Int, elapsed), remaining)

        add_log(client, user,
            "Game $i/$episodes: score=$(game.score), reached=$(1 << mt)")
    end

    # Save best game
    save_game(client, Dict(
        "name" => "$(agent_name)_test_$(length(scores))",
        "user" => user,
        "score" => Int(best_game.score),
        "numMoves" => Int(best_game.num_moves),
        "maxTile" => max_cell(best_game.board),
        "initial" => board_to_value_matrix(best_game.initial),
        "moves" => Int.(best_game.moves),
        "tiles" => [[Int(t[1]), Int(t[2]), Int(t[3])] for t in best_game.tiles]
    ))

    # Summary
    elapsed = time() - start_time
    avg = isempty(scores) ? 0 : round(Int, sum(scores) / length(scores))
    add_log(client, user,
        "Test complete: $(length(scores)) games, avg=$avg, " *
        "best=$(isempty(scores) ? 0 : maximum(scores)), " *
        "time=$(round(elapsed, digits=1))s")
end

# ============================================================================
# Watch job (real-time game replay)
# ============================================================================
function run_watch_job(client::WorkerClient, job::Any, weights_dir::String)
    description = String(job.description)
    user = String(job.user)
    agent_name = String(job.name)
    depth = Int(something(get(job, :depth, 0), 0))
    width = Int(something(get(job, :width, 1), 1))
    trigger = Int(something(get(job, :trigger_, 0), 0))

    # Load agent
    agent_info = get_agent(client, agent_name)
    if agent_info === nothing
        add_log(client, user, "Agent $agent_name not found")
        return
    end
    n = Int(agent_info.N)

    if agent_name == "Random Moves"
        evaluator = _ -> rand()
    elseif agent_name == "Best Score"
        evaluator = _ -> Float64(0)
    else
        agent = QAgent(; name=agent_name, n=n, verbose=false)
        weights_path = joinpath(weights_dir, "$agent_name.bin")
        found = download_weights(client, agent_name, weights_path)
        if found
            load_weights!(agent, weights_path)
        end
        evaluator = board -> evaluate(agent, board)
    end

    set_watch_loading(client, description, false)

    # Parse starting game state from job
    start_game = job.startGame
    initial_matrix = start_game.initial
    board = matrix_to_board_from_json(initial_matrix)
    score = Int32(start_game.score)
    num_moves = Int32(start_game.numMoves)

    # Create game record in DB
    save_game(client, Dict(
        "name" => "*$user",
        "user" => user,
        "score" => Int(score),
        "numMoves" => Int(num_moves),
        "maxTile" => max_cell(board),
        "initial" => board_to_value_matrix(board),
        "moves" => Int[],
        "tiles" => []
    ))

    rng = MersenneTwister()
    move_buffer = Int[]
    tile_buffer = Vector{Vector{Int}}()
    last_flush = time()
    flush_interval = 2.0  # seconds

    while !game_over(board)
        status = get_job_status(client, description)
        if status === nothing || status == JOB_STATUS_STOP
            break
        end

        best_dir, best_board, best_score = find_best_move(
            board, score, evaluator, depth, width, trigger, rng)

        push!(move_buffer, Int(best_dir))
        num_moves += Int32(1)
        board = best_board
        score = best_score

        # Add random tile
        empties = find_empty_cells(board)
        if !isempty(empties)
            row, col = empties[rand(rng, 1:length(empties))]
            tile = rand(rng) < 0.9 ? 1 : 2
            board = set_cell(board, row, col, tile)
            push!(tile_buffer, [Int(row), Int(col), tile])
        end

        # Flush periodically
        if time() - last_flush >= flush_interval
            update_watch_game(client, user, move_buffer, tile_buffer)
            empty!(move_buffer)
            empty!(tile_buffer)
            last_flush = time()
        end
    end

    # Game over marker
    push!(move_buffer, -1)

    # Final flush
    if !isempty(move_buffer)
        update_watch_game(client, user, move_buffer, tile_buffer)
    end

    # Keep game visible for a bit, then clean up
    sleep(10)
    delete_game(client, user)
end

# ============================================================================
# Job dispatcher
# ============================================================================
function run_job(client::WorkerClient, job::Any, weights_dir::String)
    job_type = Int(job.type)
    description = String(job.description)
    user = String(job.user)

    try
        if job_type == JOB_TYPE_TRAIN
            run_train_job(client, job, weights_dir)
        elseif job_type == JOB_TYPE_TEST
            run_test_job(client, job, weights_dir)
        elseif job_type == JOB_TYPE_WATCH
            run_watch_job(client, job, weights_dir)
        else
            add_log(client, user, "Unknown job type: $job_type")
        end
    catch e
        msg = "Job failed: $(sprint(showerror, e))"
        println(stderr, msg)
        add_log(client, "admin", "$description failed: $msg")
        add_log(client, user, "$description failed\n")
    finally
        delete_job(client, description)
    end
end
