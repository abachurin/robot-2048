using Test
using Game2048: Board, matrix_to_board, features_2, features_3, features_4, features_5, features_6,
                num_features, weight_group_sizes

@testset "Feature extraction" begin
    # Test board with known values
    m = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 0]
    board = matrix_to_board(m)

    @testset "feature counts" begin
        @test length(features_2(board)) == 24
        @test length(features_3(board)) == 52
        @test length(features_4(board)) == 17
        @test length(features_5(board)) == 21
        @test length(features_6(board)) == 33
    end

    @testset "N=4 specific values" begin
        f = features_4(board)
        # Column 0: cells (0,0)=1, (1,0)=5, (2,0)=9, (3,0)=13
        # Index = (1<<12) + (5<<8) + (9<<4) + 13 = 4096 + 1280 + 144 + 13 = 5533
        @test f[1] == (1 << 12) + (5 << 8) + (9 << 4) + 13

        # Row 0: cells (0,0)=1, (0,1)=2, (0,2)=3, (0,3)=4
        # Index = (1<<12) + (2<<8) + (3<<4) + 4 = 4096 + 512 + 48 + 4 = 4660
        @test f[5] == (1 << 12) + (2 << 8) + (3 << 4) + 4

        # Square (0,0): cells (0,0)=1, (1,0)=5, (0,1)=2, (1,1)=6
        # Index = (1<<12) + (5<<8) + (2<<4) + 6 = 4096 + 1280 + 32 + 6 = 5414
        @test f[9] == (1 << 12) + (5 << 8) + (2 << 4) + 6
    end

    @testset "N=5 includes N=4 plus cross features" begin
        f4 = features_4(board)
        f5 = features_5(board)
        # First 17 should match N=4
        for i in 1:17
            @test f5[i] == f4[i]
        end
        # Cross at (1,1): center=6, above=2, left=5, below=10, right=7
        # Index = (6<<16) + (2<<12) + (5<<8) + (10<<4) + 7 = 393216+8192+1280+160+7
        @test f5[18] == (6 << 16) + (2 << 12) + (5 << 8) + (10 << 4) + 7
    end

    @testset "N=6 includes N=5 plus 6-tuples" begin
        f5 = features_5(board)
        f6 = features_6(board)
        for i in 1:21
            @test f6[i] == f5[i]
        end
        # 6-tuple features should be non-negative
        for i in 22:33
            @test f6[i] >= 0
        end
    end

    @testset "features within weight bounds" begin
        for (n, ff) in [(2, features_2), (3, features_3), (4, features_4),
                        (5, features_5), (6, features_6)]
            feats = ff(board)
            sizes = weight_group_sizes(n)
            for (g, idx) in enumerate(feats)
                @test 0 <= idx < sizes[g]
            end
        end
    end

    @testset "zero board features" begin
        zb = Board(0)
        # All features should be 0 for zero board
        for f in features_4(zb)
            @test f == 0
        end
    end
end
