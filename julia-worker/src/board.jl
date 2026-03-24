# Board representation: UInt64 with 16 cells × 4 bits each
# Cell (i,j) where i=row (0-3), j=col (0-3) is at bit offset (15 - 4i - j) * 4
# Row 0 = bits 48-63 (most significant), Row 3 = bits 0-15 (least significant)
# Cell values: 0 = empty, k = tile 2^k (1=2, 2=4, ..., 15=32768)

const Board = UInt64
const CELL_MASK = UInt64(0xF)
const ROW_MASK = UInt64(0xFFFF)

@inline function cell_shift(row::Int, col::Int)::Int
    (15 - 4 * row - col) << 2
end

@inline function get_cell(board::Board, row::Int, col::Int)::Int
    Int((board >> cell_shift(row, col)) & CELL_MASK)
end

@inline function set_cell(board::Board, row::Int, col::Int, val::Int)::Board
    s = cell_shift(row, col)
    (board & ~(CELL_MASK << s)) | (UInt64(val) << s)
end

@inline function get_row(board::Board, row::Int)::UInt16
    UInt16((board >> ((3 - row) << 4)) & ROW_MASK)
end

@inline function set_row(board::Board, row::Int, rowval::UInt16)::Board
    s = (3 - row) << 4
    (board & ~(ROW_MASK << s)) | (UInt64(rowval) << s)
end

@inline function extract_col(board::Board, col::Int)::UInt16
    c0 = (board >> cell_shift(0, col)) & CELL_MASK
    c1 = (board >> cell_shift(1, col)) & CELL_MASK
    c2 = (board >> cell_shift(2, col)) & CELL_MASK
    c3 = (board >> cell_shift(3, col)) & CELL_MASK
    UInt16((c0 << 12) | (c1 << 8) | (c2 << 4) | c3)
end

function set_col(board::Board, col::Int, colval::UInt16)::Board
    for row in 0:3
        val = Int((colval >> ((3 - row) << 2)) & 0xF)
        board = set_cell(board, row, col, val)
    end
    board
end

function transpose_board(board::Board)::Board
    (UInt64(extract_col(board, 0)) << 48) |
    (UInt64(extract_col(board, 1)) << 32) |
    (UInt64(extract_col(board, 2)) << 16) |
    UInt64(extract_col(board, 3))
end

# Counterclockwise 90°: result[i,j] = board[j, 3-i]
# Row i of result = column (3-i) of board
function rot90_ccw(board::Board)::Board
    (UInt64(extract_col(board, 3)) << 48) |
    (UInt64(extract_col(board, 2)) << 32) |
    (UInt64(extract_col(board, 1)) << 16) |
    UInt64(extract_col(board, 0))
end

# Clockwise 90°: result[i,j] = board[3-j, i]
function rot90_cw(board::Board)::Board
    rot90_ccw(rot90_ccw(rot90_ccw(board)))
end

# 180° rotation
function rot180(board::Board)::Board
    rot90_ccw(rot90_ccw(board))
end

# Count empty cells
function empty_count(board::Board)::Int
    count = 0
    b = board
    for _ in 1:16
        if (b & CELL_MASK) == 0
            count += 1
        end
        b >>= 4
    end
    count
end

# Find empty cell positions as (row, col) tuples
function find_empty_cells(board::Board)::Vector{Tuple{Int,Int}}
    cells = Tuple{Int,Int}[]
    for i in 0:3, j in 0:3
        if get_cell(board, i, j) == 0
            push!(cells, (i, j))
        end
    end
    cells
end

# Max cell value on the board
function max_cell(board::Board)::Int
    m = 0
    b = board
    for _ in 1:16
        v = Int(b & CELL_MASK)
        if v > m
            m = v
        end
        b >>= 4
    end
    m
end

# Convert board to 4x4 matrix (for display/serialization)
function board_to_matrix(board::Board)::Matrix{Int}
    m = Matrix{Int}(undef, 4, 4)
    for i in 0:3, j in 0:3
        m[i+1, j+1] = get_cell(board, i, j)
    end
    m
end

# Convert 4x4 matrix to board
function matrix_to_board(m::Matrix{Int})::Board
    board = Board(0)
    for i in 0:3, j in 0:3
        board = set_cell(board, i, j, m[i+1, j+1])
    end
    board
end

# Pretty-print board with tile values
function print_board(io::IO, board::Board)
    for i in 0:3
        for j in 0:3
            v = get_cell(board, i, j)
            tile = v == 0 ? 0 : 1 << v
            print(io, lpad(tile, 6))
        end
        println(io)
    end
end
print_board(board::Board) = print_board(stdout, board)
