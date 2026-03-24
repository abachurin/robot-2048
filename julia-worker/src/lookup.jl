# Pre-computed move lookup tables
# Two flat arrays of 65536 entries each, indexed by UInt16 row value + 1

struct RowResult
    new_row::UInt16
    score::Int32
end

# Tables: indexed by Int(row) + 1 (Julia 1-based)
const MOVE_LEFT = Vector{RowResult}(undef, 65536)
const MOVE_RIGHT = Vector{RowResult}(undef, 65536)

# Whether a left-move changes this row
const CHANGED_LEFT = Vector{Bool}(undef, 65536)
const CHANGED_RIGHT = Vector{Bool}(undef, 65536)

function generate_tables!()
    # Pass 1: compute all MOVE_LEFT entries
    for packed_int in 0:65535
        packed = UInt16(packed_int)
        a = Int((packed >> 12) & 0xF)
        b = Int((packed >> 8) & 0xF)
        cc = Int((packed >> 4) & 0xF)
        d = Int(packed & 0xF)

        # Slide non-zero cells left, merge adjacent equals
        cells = Int[]
        for v in (a, b, cc, d)
            v != 0 && push!(cells, v)
        end

        score = Int32(0)
        i = 1
        while i < length(cells)
            if cells[i] == cells[i+1]
                cells[i] = min(cells[i] + 1, 15)  # cap at 15 (4 bits)
                score += Int32(1) << cells[i]
                deleteat!(cells, i + 1)
            end
            i += 1
        end
        while length(cells) < 4
            push!(cells, 0)
        end

        new_packed = UInt16((cells[1] << 12) | (cells[2] << 8) | (cells[3] << 4) | cells[4])
        idx = Int(packed) + 1
        MOVE_LEFT[idx] = RowResult(new_packed, score)
        CHANGED_LEFT[idx] = new_packed != packed
    end

    # Pass 2: compute MOVE_RIGHT from MOVE_LEFT (all LEFT entries now available)
    for packed_int in 0:65535
        packed = UInt16(packed_int)
        a = Int((packed >> 12) & 0xF)
        b = Int((packed >> 8) & 0xF)
        cc = Int((packed >> 4) & 0xF)
        d = Int(packed & 0xF)

        rev_packed = UInt16((d << 12) | (cc << 8) | (b << 4) | a)
        rev_result = MOVE_LEFT[Int(rev_packed) + 1]
        ra = Int((rev_result.new_row >> 12) & 0xF)
        rb = Int((rev_result.new_row >> 8) & 0xF)
        rc = Int((rev_result.new_row >> 4) & 0xF)
        rd = Int(rev_result.new_row & 0xF)
        right_row = UInt16((rd << 12) | (rc << 8) | (rb << 4) | ra)
        idx = Int(packed) + 1
        MOVE_RIGHT[idx] = RowResult(right_row, rev_result.score)
        CHANGED_RIGHT[idx] = right_row != packed
    end
end

# Direction constants matching Python: 0=left, 1=up, 2=right, 3=down
const DIR_LEFT = 0
const DIR_UP = 1
const DIR_RIGHT = 2
const DIR_DOWN = 3

# Move the board in a direction. Returns (new_board, score_gained, changed)
function make_move(board::Board, direction::Int)::Tuple{Board, Int32, Bool}
    if direction == DIR_LEFT
        return move_left(board)
    elseif direction == DIR_RIGHT
        return move_right(board)
    elseif direction == DIR_UP
        return move_up(board)
    else
        return move_down(board)
    end
end

function move_left(board::Board)::Tuple{Board, Int32, Bool}
    new_board = Board(0)
    total_score = Int32(0)
    changed = false
    for r in 0:3
        row = get_row(board, r)
        idx = Int(row) + 1
        @inbounds result = MOVE_LEFT[idx]
        @inbounds changed |= CHANGED_LEFT[idx]
        new_board = set_row(new_board, r, result.new_row)
        total_score += result.score
    end
    (new_board, total_score, changed)
end

function move_right(board::Board)::Tuple{Board, Int32, Bool}
    new_board = Board(0)
    total_score = Int32(0)
    changed = false
    for r in 0:3
        row = get_row(board, r)
        idx = Int(row) + 1
        @inbounds result = MOVE_RIGHT[idx]
        @inbounds changed |= CHANGED_RIGHT[idx]
        new_board = set_row(new_board, r, result.new_row)
        total_score += result.score
    end
    (new_board, total_score, changed)
end

function move_up(board::Board)::Tuple{Board, Int32, Bool}
    new_board = Board(0)
    total_score = Int32(0)
    changed = false
    for c in 0:3
        col = extract_col(board, c)
        idx = Int(col) + 1
        @inbounds result = MOVE_LEFT[idx]
        @inbounds changed |= CHANGED_LEFT[idx]
        new_board = set_col(new_board, c, result.new_row)
        total_score += result.score
    end
    (new_board, total_score, changed)
end

function move_down(board::Board)::Tuple{Board, Int32, Bool}
    new_board = Board(0)
    total_score = Int32(0)
    changed = false
    for c in 0:3
        col = extract_col(board, c)
        idx = Int(col) + 1
        @inbounds result = MOVE_RIGHT[idx]
        @inbounds changed |= CHANGED_RIGHT[idx]
        new_board = set_col(new_board, c, result.new_row)
        total_score += result.score
    end
    (new_board, total_score, changed)
end
