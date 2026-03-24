using Test
using Game2048: Board, matrix_to_board, d4_symmetries, get_cell

@testset "D4 symmetry" begin
    # Use a board with no symmetry so all 8 transforms are distinct
    m = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 0]
    board = matrix_to_board(m)

    syms = d4_symmetries(board)

    @testset "produces 8 distinct boards" begin
        @test length(Set(syms)) == 8
    end

    @testset "original is first" begin
        @test syms[1] == board
    end

    @testset "closure — applying d4 to any element gives same set" begin
        for s in syms
            other_syms = Set(d4_symmetries(s))
            @test other_syms == Set(syms)
        end
    end
end
