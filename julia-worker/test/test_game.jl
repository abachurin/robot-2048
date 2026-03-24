using Test
using Random
using Game2048: Board, new_game, game_over, empty_count, matrix_to_board, add_random_tile!, max_cell

@testset "Game logic" begin
    @testset "new_game" begin
        rng = MersenneTwister(42)
        game = new_game(rng)
        # Should have exactly 2 tiles placed
        @test empty_count(game.board) == 14
        # Initial should match current board
        @test game.initial == game.board
        # Score and moves should be 0
        @test game.score == 0
        @test game.num_moves == 0
        @test isempty(game.moves)
        @test isempty(game.tiles)  # initial tiles cleared
    end

    @testset "game_over detection" begin
        # Empty board — not over
        @test !game_over(Board(0))

        # Full board with no merges — game over
        b = matrix_to_board([1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 1])
        @test game_over(b)

        # Full board with adjacent equal — not over
        b = matrix_to_board([1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 15])
        @test !game_over(b)
    end

    @testset "add_random_tile" begin
        rng = MersenneTwister(123)
        game = new_game(rng)
        ec_before = empty_count(game.board)
        add_random_tile!(game, rng)
        @test empty_count(game.board) == ec_before - 1
        # Tile should be 1 or 2
        last_tile = game.tiles[end]
        @test last_tile[3] in (Int8(1), Int8(2))
    end

    @testset "deterministic with same seed" begin
        g1 = new_game(MersenneTwister(999))
        g2 = new_game(MersenneTwister(999))
        @test g1.board == g2.board
    end
end
