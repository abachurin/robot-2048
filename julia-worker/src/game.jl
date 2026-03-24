# Game state and game loop

mutable struct GameState
    board::Board
    score::Int32
    num_moves::Int32
    moves::Vector{Int8}
    tiles::Vector{Tuple{Int8, Int8, Int8}}  # (row, col, tile_value)
    initial::Board
end

function new_game(rng::AbstractRNG=Random.default_rng())::GameState
    game = GameState(Board(0), Int32(0), Int32(0), Int8[], Tuple{Int8,Int8,Int8}[], Board(0))
    add_random_tile!(game, rng)
    add_random_tile!(game, rng)
    game.tiles = Tuple{Int8,Int8,Int8}[]  # clear initial tiles from recording (matches Python)
    game.initial = game.board
    game
end

function add_random_tile!(game::GameState, rng::AbstractRNG=Random.default_rng())
    empties = find_empty_cells(game.board)
    isempty(empties) && return
    row, col = empties[rand(rng, 1:length(empties))]
    tile = rand(rng) < 0.9 ? Int8(1) : Int8(2)
    game.board = set_cell(game.board, row, col, Int(tile))
    push!(game.tiles, (Int8(row), Int8(col), tile))
end

function game_over(board::Board)::Bool
    # No empty cells?
    empty_count(board) > 0 && return false
    # Any valid move?
    for dir in 0:3
        _, _, changed = make_move(board, dir)
        changed && return false
    end
    true
end

# Lookahead search for test/watch mode
function look_forward(
    board::Board, score::Int32, evaluator,
    depth::Int, width::Int, trigger::Int,
    rng::AbstractRNG
)::Float64
    if depth == 0
        return evaluator(board)
    end
    ec = empty_count(board)
    if ec > trigger
        return evaluator(board)
    end
    empties = find_empty_cells(board)
    num_tiles = min(width, length(empties))
    positions = randperm(rng, length(empties))[1:num_tiles]

    average = Float64(0)
    for pos_idx in positions
        row, col = empties[pos_idx]
        tile = rand(rng) < 0.9 ? 1 : 2
        new_board = set_cell(board, row, col, tile)
        if game_over(new_board)
            best_value = Float64(0)
        else
            best_value = -Inf32
            for dir in 0:3
                test_board, move_score, changed = make_move(new_board, dir)
                if changed
                    value = look_forward(test_board, score + move_score, evaluator,
                                         depth - 1, width, trigger, rng)
                    best_value = max(best_value, value)
                end
            end
        end
        average += max(best_value, Float64(0))
    end
    average / num_tiles
end

function find_best_move(
    board::Board, score::Int32, evaluator,
    depth::Int, width::Int, trigger::Int,
    rng::AbstractRNG
)::Tuple{Int8, Board, Int32}
    best_dir = Int8(0)
    best_value = -Inf32
    best_board = Board(0)
    best_score = Int32(0)
    for dir in 0:3
        new_board, move_score, changed = make_move(board, dir)
        if changed
            new_score = score + move_score
            value = look_forward(new_board, new_score, evaluator,
                                  depth, width, trigger, rng)
            if value > best_value
                best_dir = Int8(dir)
                best_value = value
                best_board = new_board
                best_score = new_score
            end
        end
    end
    (best_dir, best_board, best_score)
end

# Run a test episode (no learning, with lookahead)
function trial_run!(game::GameState, evaluator; depth::Int=0, width::Int=1, trigger::Int=0,
                    rng::AbstractRNG=Random.default_rng())
    while !game_over(game.board)
        best_dir, best_board, best_score = find_best_move(
            game.board, game.score, evaluator, depth, width, trigger, rng)
        push!(game.moves, best_dir)
        game.num_moves += Int32(1)
        game.board = best_board
        game.score = best_score
        add_random_tile!(game, rng)
    end
end

# Pretty-print game state
function Base.show(io::IO, game::GameState)
    print_board(io, game.board)
    println(io, "score = $(game.score), moves = $(game.num_moves), reached $(1 << max_cell(game.board))")
end
