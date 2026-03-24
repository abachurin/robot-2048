# RL Agent — TD(0) value learning with N-tuple features and D4 symmetry
#
# Key properties:
# 1) gamma = 1 (no discounting — episodic task)
# 2) epsilon = 0 (greedy — game is stochastic enough)
# 3) Linear value function: V(s) = sum of weights at active features
# 4) Sparse update: only num_features weights per state, ×8 for D4 symmetry

# Learning rate modes:
# :global_decay  — classic: decay alpha on a schedule (original Python behavior)
# :constant      — fixed alpha, no decay
# :frontier      — decay per-update based on how far board's max tile is below frontier
@enum AlphaMode GLOBAL_DECAY CONSTANT FRONTIER

mutable struct QAgent{T <: AbstractFloat}
    name::String
    n::Int
    weights::WeightTable{T}
    feature_fn::Function
    nf::Int                      # number of feature groups
    alpha::Float64               # learning rate (base)
    decay::Float64               # decay multiplier
    step::Int                    # for GLOBAL_DECAY: decay every this many episodes
    min_alpha::Float64           # LR floor
    alpha_mode::AlphaMode
    frontier::Int                # highest max_tile seen (for FRONTIER mode)
    max_tile::Int
    best_score::Int32
    last_episode::Int
    history::Vector{Int}         # training history (moving averages)
    collect_step::Int            # history sampling interval
    next_decay::Int
    max_train_history::Int
end

function QAgent(;
    name::String = "agent",
    n::Int = 4,
    alpha::Float64 = 0.25,
    decay::Float64 = 0.5,
    step::Int = 1000,
    min_alpha::Float64 = 0.01,
    alpha_mode::AlphaMode = GLOBAL_DECAY,
    float_type::Type{T} = Float32,
    transfer::Bool = false,
    verbose::Bool = true,
) where T <: AbstractFloat
    feature_fn = get_feature_function(n)
    nf = num_features(n)
    sizes = weight_group_sizes(n)

    if transfer
        bases = weight_group_bases(n)
        tsizes = weight_group_tuple_sizes(n)
        weights = WeightTable{T}(sizes; enable_transfer=true, group_bases=bases, group_tuple_sizes=tsizes)
    else
        weights = WeightTable{T}(sizes)
    end

    mode_str = alpha_mode == GLOBAL_DECAY ? "global_decay(step=$step)" :
               alpha_mode == CONSTANT ? "constant" : "frontier(decay=$decay)"

    if verbose
        println("Created agent '$name' with N=$n ($T)")
        println("  Features: $nf groups")
        println("  Weights: $(length(weights.data)) entries ($(round(memory_mb(weights), digits=1)) MB)")
        transfer && println("  Transfer: enabled (+$(round(transfer_overhead_mb(weights), digits=1)) MB overhead)")
        println("  Alpha: $alpha, mode: $mode_str")
    end

    QAgent{T}(name, n, weights, feature_fn, nf,
              alpha, decay, step, min_alpha, alpha_mode,
              0,  # frontier
              0, Int32(0), 0, Int[], 1000, step, 200)
end

# Evaluate a board position
@inline function evaluate(agent::QAgent{T}, board::Board)::Float64 where T
    evaluate(agent.weights, agent.feature_fn(board))
end

# Update weights for a board across all 8 D4 symmetries
function update!(agent::QAgent{T}, board::Board, dw::Float64) where T
    for sym_board in d4_symmetries(board)
        update_weights!(agent.weights, agent.feature_fn(sym_board), dw)
    end
end

# Compute effective alpha for a given board state
@inline function effective_alpha(agent::QAgent, board_max::Int)::Float64
    if agent.alpha_mode == FRONTIER
        gap = agent.frontier - board_max
        if gap <= 1
            agent.alpha  # full alpha at frontier and one level below
        else
            agent.alpha * agent.decay ^ (gap - 1)
        end
    else
        agent.alpha
    end
end

# Single training episode — plays one game with TD(0) updates
function episode!(agent::QAgent, rng::AbstractRNG=Random.default_rng())::GameState
    game = new_game(rng)
    state = Board(0)
    old_label = Float64(0)
    first = true
    state_max = 0  # max_cell of the state we're about to update

    while !game_over(game.board)
        best_dir = Int8(0)
        best_value = -Inf
        best_board = Board(0)
        best_score = Int32(0)

        for dir in 0:3
            new_board, move_score, changed = make_move(game.board, dir)
            if changed
                value = evaluate(agent, new_board)
                if value > best_value
                    best_dir = Int8(dir)
                    best_value = value
                    best_board = new_board
                    best_score = game.score + move_score
                end
            end
        end

        if !first
            alpha = effective_alpha(agent, state_max)
            reward = Float64(best_score - game.score)
            dw = (reward + best_value - old_label) * alpha / agent.nf
            update!(agent, state, dw)
        end
        first = false

        push!(game.moves, best_dir)
        game.num_moves += Int32(1)
        game.board = best_board
        game.score = best_score
        state = game.board
        state_max = max_cell(game.board)
        old_label = best_value

        add_random_tile!(game, rng)
    end

    # Terminal update
    push!(game.moves, Int8(-1))
    alpha = effective_alpha(agent, state_max)
    dw = -old_label * alpha / agent.nf
    update!(agent, state, dw)

    # Update frontier
    game_max = max_cell(game.board)
    agent.frontier = max(agent.frontier, game_max)

    game
end

# Decay learning rate (for GLOBAL_DECAY mode)
function decay_alpha!(agent::QAgent)
    agent.alpha = max(agent.alpha * agent.decay, agent.min_alpha)
    agent.next_decay = agent.last_episode + agent.step
end

# ============================================================================
# Training loop
# ============================================================================
function train!(agent::QAgent, episodes::Int; rng::AbstractRNG=Random.default_rng())
    start_time = time()
    start_1000 = start_time
    first_episode = agent.last_episode
    last_episode = agent.last_episode + episodes

    ma100 = Float64[]
    av1000 = Float64[]
    ma_collect = Float64[]
    reached = zeros(Int, 7)   # indices for tiles 1024..65536
    best_of_1000 = new_game(rng)

    println("\n$(agent.name): training $episodes episodes (from $(first_episode + 1) to $last_episode)")
    println("  Weights: $(round(memory_mb(agent.weights), digits=1)) MB\n")

    while agent.last_episode < last_episode
        game = episode!(agent, rng)
        agent.last_episode += 1
        push!(ma100, game.score)
        push!(av1000, game.score)
        mt = max_cell(game.board)

        if game.score > best_of_1000.score
            best_of_1000 = game
            if game.score > agent.best_score
                agent.best_score = game.score
                println("\n  New best score at episode $(agent.last_episode): $(game.score), " *
                        "reached $(1 << mt)")
            end
        end

        if mt >= 10
            reached[min(mt - 9, 7)] += 1
        end
        agent.max_tile = max(agent.max_tile, mt)

        # Log every 100 episodes
        if agent.last_episode % 100 == 0
            average = sum(ma100) / length(ma100)
            push!(ma_collect, average)
            println("  episode $(agent.last_episode), last 100 avg = $(round(Int, average))")
            empty!(ma100)

            if agent.last_episode % agent.collect_step == 0
                push!(agent.history, round(Int, sum(ma_collect) / length(ma_collect)))
                empty!(ma_collect)
                if length(agent.history) >= agent.max_train_history
                    agent.history = agent.history[1:2:end]
                    agent.collect_step *= 2
                end
            end
        end

        # Summary every 1000 episodes
        if agent.last_episode % 1000 == 0
            avg = round(Int, sum(av1000) / length(av1000))
            elapsed_1000 = time() - start_1000
            total_elapsed = time() - start_time
            eps_done = agent.last_episode - first_episode
            eps_remaining = last_episode - agent.last_episode
            eta = eps_remaining > 0 ? round(Int, total_elapsed / eps_done * eps_remaining) : 0

            println("\n  === Episode $(agent.last_episode) ===")
            println("  $(round(elapsed_1000, digits=1))s for last $(length(av1000)) episodes")
            println("  Average score: $avg")
            for j in 1:7
                r = sum(reached[j:7]) / length(av1000) * 100
                r > 0 && println("  $(1 << (j + 9)) reached: $(round(r, digits=2))%")
            end
            println("  Best of batch: score=$(best_of_1000.score), reached $(1 << max_cell(best_of_1000.board))")
            println("  Best ever: score=$(agent.best_score), frontier=$(1 << agent.frontier)")
            eta > 0 && println("  ETA: $(div(eta, 60)) min $(eta % 60) sec")
            println()

            empty!(av1000)
            fill!(reached, 0)
            best_of_1000 = new_game(rng)
            start_1000 = time()
        end

        # Learning rate decay (GLOBAL_DECAY mode only)
        if agent.alpha_mode == GLOBAL_DECAY
            if agent.last_episode >= agent.next_decay && agent.alpha > agent.min_alpha
                decay_alpha!(agent)
                println("  LR decayed to $(round(agent.alpha, digits=6))")
            end
        end
    end

    total = time() - start_time
    println("Training complete. Total time: $(round(total, digits=1))s " *
            "($(round(episodes / total, digits=0)) episodes/sec)")
end

# ============================================================================
# Test run (with lookahead)
# ============================================================================
function test_run(agent::QAgent, episodes::Int;
                  depth::Int=0, width::Int=1, trigger::Int=0,
                  rng::AbstractRNG=Random.default_rng())
    println("\n$(agent.name): testing $episodes episodes")
    println("  Lookahead: depth=$depth, width=$width, trigger=$trigger\n")

    results = Tuple{Int32, Int, Int32}[]  # (score, max_tile, num_moves)
    start = time()

    for i in 1:episodes
        game = new_game(rng)
        evaluator = board -> evaluate(agent, board)
        trial_run!(game, evaluator; depth, width, trigger, rng)

        mt = max_cell(game.board)
        push!(results, (game.score, mt, game.num_moves))
        t = time() - start
        println("  game $i: score=$(game.score), moves=$(game.num_moves), " *
                "reached $(1 << mt), time=$(round(t / i, digits=2))s/game")
    end

    scores = [r[1] for r in results]
    tiles = [r[2] for r in results]
    total_moves = sum(r[3] for r in results)
    elapsed = time() - start

    println("\n  === Test Results ($episodes games) ===")
    println("  Average score: $(round(Int, sum(scores) / length(scores)))")
    for thresh in [14, 13, 12, 11, 10]  # 16384, 8192, 4096, 2048, 1024
        pct = count(t -> t >= thresh, tiles) / length(tiles) * 100
        pct > 0 && println("  $(1 << thresh) reached: $(round(pct, digits=1))%")
    end
    println("  Avg time/move: $(round(elapsed / total_moves * 1000, digits=2)) ms")
    println("  Total time: $(round(elapsed, digits=1))s")
end

# ============================================================================
# Save/Load weights
# ============================================================================
function save_weights(path::String, agent::QAgent)
    open(path, "w") do io
        write(io, Int32(agent.n))
        write(io, Int32(agent.last_episode))
        write(io, Int32(agent.best_score))
        write(io, Int32(agent.max_tile))
        write(io, Float64(agent.alpha))
        write(io, Int32(length(agent.weights.data)))
        write(io, agent.weights.data)
    end
    mb = filesize(path) / (1024 * 1024)
    println("Weights saved to $path ($(round(mb, digits=1)) MB)")
end

function load_weights!(agent::QAgent, path::String)
    open(path, "r") do io
        n = read(io, Int32)
        n != agent.n && error("Weight file N=$n doesn't match agent N=$(agent.n)")
        agent.last_episode = read(io, Int32)
        agent.best_score = read(io, Int32)
        agent.max_tile = read(io, Int32)
        agent.alpha = read(io, Float64)
        agent.next_decay = agent.last_episode + agent.step
        len = read(io, Int32)
        len != length(agent.weights.data) && error("Weight count mismatch: file=$len, agent=$(length(agent.weights.data))")
        read!(io, agent.weights.data)
    end
    println("Weights loaded from $path")
    println("  N=$(agent.n), episodes=$(agent.last_episode), best=$(agent.best_score), " *
            "alpha=$(round(agent.alpha, digits=6))")
end
