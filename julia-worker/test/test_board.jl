using Test
using Game2048: Board, get_cell, set_cell, get_row, set_row, extract_col, set_col,
                transpose_board, rot90_ccw, empty_count, max_cell, board_to_matrix,
                matrix_to_board, print_board

@testset "Board operations" begin
    # Build a known board:
    # Row 0: [1, 2, 3, 4]
    # Row 1: [5, 6, 7, 8]
    # Row 2: [9, 10, 11, 12]
    # Row 3: [13, 14, 15, 0]
    m = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 0]
    board = matrix_to_board(m)

    @testset "get_cell / set_cell" begin
        for i in 0:3, j in 0:3
            @test get_cell(board, i, j) == m[i+1, j+1]
        end
        # Set and read back
        b2 = set_cell(board, 3, 3, 5)
        @test get_cell(b2, 3, 3) == 5
        @test get_cell(b2, 0, 0) == 1  # unchanged
    end

    @testset "get_row / set_row" begin
        # Row 0 should be 0x1234
        @test get_row(board, 0) == UInt16(0x1234)
        @test get_row(board, 1) == UInt16(0x5678)
        @test get_row(board, 3) == UInt16(0xDEF0)
        # Set row and verify
        b2 = set_row(board, 2, UInt16(0xAAAA))
        @test get_row(b2, 2) == UInt16(0xAAAA)
        @test get_row(b2, 0) == UInt16(0x1234)  # unchanged
    end

    @testset "extract_col / set_col" begin
        # Column 0 should be [1, 5, 9, 13] = 0x159D
        @test extract_col(board, 0) == UInt16(0x159D)
        @test extract_col(board, 3) == UInt16(0x48C0)
        # Round-trip
        b2 = set_col(Board(0), 1, extract_col(board, 1))
        for i in 0:3
            @test get_cell(b2, i, 1) == get_cell(board, i, 1)
        end
    end

    @testset "transpose" begin
        tb = transpose_board(board)
        for i in 0:3, j in 0:3
            @test get_cell(tb, i, j) == get_cell(board, j, i)
        end
        # Double transpose = identity
        @test transpose_board(tb) == board
    end

    @testset "rot90" begin
        # rot90_ccw: result[i,j] = board[j, 3-i]
        rb = rot90_ccw(board)
        for i in 0:3, j in 0:3
            @test get_cell(rb, i, j) == get_cell(board, j, 3 - i)
        end
        # 4 rotations = identity
        @test rot90_ccw(rot90_ccw(rot90_ccw(rb))) == board
    end

    @testset "empty_count / max_cell" begin
        @test empty_count(board) == 1  # only (3,3) is 0
        @test max_cell(board) == 15
        @test empty_count(Board(0)) == 16
        @test max_cell(Board(0)) == 0
    end

    @testset "matrix round-trip" begin
        @test board_to_matrix(board) == m
        @test matrix_to_board(board_to_matrix(board)) == board
    end
end
