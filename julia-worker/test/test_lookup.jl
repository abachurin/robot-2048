using Test
using Game2048: Board, matrix_to_board, board_to_matrix, make_move, get_cell,
                DIR_LEFT, DIR_RIGHT, DIR_UP, DIR_DOWN

@testset "Lookup tables and moves" begin
    @testset "move_left basic cases" begin
        # [2, 2, 0, 0] → [4, 0, 0, 0], score = 4
        b = matrix_to_board([1 1 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0])
        nb, score, changed = make_move(b, DIR_LEFT)
        @test get_cell(nb, 0, 0) == 2  # merged to 4
        @test get_cell(nb, 0, 1) == 0
        @test score == 4
        @test changed

        # [2, 0, 0, 2] → [4, 0, 0, 0]
        b = matrix_to_board([1 0 0 1; 0 0 0 0; 0 0 0 0; 0 0 0 0])
        nb, score, changed = make_move(b, DIR_LEFT)
        @test get_cell(nb, 0, 0) == 2
        @test score == 4
        @test changed

        # [2, 2, 2, 2] → [4, 4, 0, 0], score = 8
        b = matrix_to_board([1 1 1 1; 0 0 0 0; 0 0 0 0; 0 0 0 0])
        nb, score, changed = make_move(b, DIR_LEFT)
        @test get_cell(nb, 0, 0) == 2
        @test get_cell(nb, 0, 1) == 2
        @test get_cell(nb, 0, 2) == 0
        @test score == 8
        @test changed

        # [4, 2, 2, 0] → [4, 4, 0, 0], score = 4
        b = matrix_to_board([2 1 1 0; 0 0 0 0; 0 0 0 0; 0 0 0 0])
        nb, score, changed = make_move(b, DIR_LEFT)
        @test get_cell(nb, 0, 0) == 2
        @test get_cell(nb, 0, 1) == 2
        @test score == 4
        @test changed
    end

    @testset "no change" begin
        # [2, 4, 8, 16] left → no change
        b = matrix_to_board([1 2 3 4; 0 0 0 0; 0 0 0 0; 0 0 0 0])
        _, _, changed = make_move(b, DIR_LEFT)
        @test !changed

        # empty board
        _, _, changed = make_move(Board(0), DIR_LEFT)
        @test !changed
    end

    @testset "move_right" begin
        # [2, 2, 0, 0] right → [0, 0, 0, 4]
        b = matrix_to_board([1 1 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0])
        nb, score, changed = make_move(b, DIR_RIGHT)
        @test get_cell(nb, 0, 3) == 2
        @test get_cell(nb, 0, 2) == 0
        @test score == 4
        @test changed
    end

    @testset "move_up" begin
        # Column with [2, 2, 0, 0] up → [4, 0, 0, 0]
        b = matrix_to_board([1 0 0 0; 1 0 0 0; 0 0 0 0; 0 0 0 0])
        nb, score, changed = make_move(b, DIR_UP)
        @test get_cell(nb, 0, 0) == 2
        @test get_cell(nb, 1, 0) == 0
        @test score == 4
        @test changed
    end

    @testset "move_down" begin
        # Column with [2, 2, 0, 0] down → [0, 0, 0, 4]
        b = matrix_to_board([1 0 0 0; 1 0 0 0; 0 0 0 0; 0 0 0 0])
        nb, score, changed = make_move(b, DIR_DOWN)
        @test get_cell(nb, 3, 0) == 2
        @test get_cell(nb, 2, 0) == 0
        @test score == 4
        @test changed
    end

    @testset "multi-row move" begin
        # Full board move
        b = matrix_to_board([1 1 2 2; 3 3 0 0; 0 0 4 4; 1 0 0 1])
        nb, score, changed = make_move(b, DIR_LEFT)
        m = board_to_matrix(nb)
        @test m[1, :] == [2, 3, 0, 0]  # 2+2=4, 4+4=8
        @test m[2, :] == [4, 0, 0, 0]  # 8+8=16
        @test m[3, :] == [5, 0, 0, 0]  # 16+16=32
        @test m[4, :] == [2, 0, 0, 0]  # 2+2=4
        @test score == 4 + 8 + 16 + 32 + 4
        @test changed
    end
end
