using Test
using Random
using Game2048: QAgent, episode!, evaluate, train!, Board, matrix_to_board, memory_mb

@testset "Agent" begin
    @testset "agent creation" begin
        agent = QAgent(name="test", n=4, alpha=0.01)
        @test agent.n == 4
        @test agent.nf == 17
        @test length(agent.weights.data) == 17 * 65536
        @test memory_mb(agent.weights) ≈ 17 * 65536 * 4 / (1024 * 1024)
    end

    @testset "evaluate returns finite value" begin
        agent = QAgent(name="test", n=4)
        board = matrix_to_board([1 2 0 0; 0 3 0 0; 0 0 0 0; 0 0 0 0])
        v = evaluate(agent, board)
        @test isfinite(v)
    end

    @testset "episode produces valid game" begin
        rng = MersenneTwister(42)
        agent = QAgent(name="test", n=4)
        game = episode!(agent, rng)
        @test game.score >= 0
        @test game.num_moves > 0
        @test game.moves[end] == Int8(-1)  # terminal marker
        @test length(game.moves) == game.num_moves + 1
    end

    @testset "training changes weights" begin
        rng = MersenneTwister(42)
        agent = QAgent(name="test", n=4, alpha=0.01)
        w_before = copy(agent.weights.data)
        episode!(agent, rng)
        @test agent.weights.data != w_before
    end

    @testset "short training run" begin
        agent = QAgent(name="test", n=4, alpha=0.005, step=500)
        rng = MersenneTwister(42)
        train!(agent, 200; rng)
        @test agent.last_episode == 200
        @test agent.best_score > 0
    end

    @testset "N=5 creates without error" begin
        a5 = QAgent(name="test5", n=5)
        @test a5.nf == 21
        # N=6 skipped in tests — 365MB allocation too slow for CI
    end
end
