# N-tuple feature extraction
# Each function returns a tuple of weight indices, matching Python f_2..f_6 exactly.
# All ravel order is row-major (C order) to match numpy.

# Helper: extract all 16 cells into a tuple for fast repeated access
@inline function all_cells(board::Board)
    # c[i*4+j+1] = cell(i,j) for i,j in 0:3
    ntuple(k -> Int((board >> ((16 - k) << 2)) & 0xF), Val(16))
end

# Convenience: c(cells, i, j) = cells[4i + j + 1]  (0-indexed i,j)
@inline c(cells, i, j) = @inbounds cells[4i + j + 1]

# ============================================================================
# N=2: all adjacent pairs (24 features, each in 0..255)
# x_vert: (3,4) array → 12 values, x_hor: (4,3) array → 12 values
# ============================================================================
function features_2(board::Board)::NTuple{24, Int}
    cc = all_cells(board)
    # Vertical pairs: (i,j)&(i+1,j) for i=0:2, j=0:3, ravel (3,4) row-major
    # Horizontal pairs: (i,j)&(i,j+1) for i=0:3, j=0:2, ravel (4,3) row-major
    ntuple(Val(24)) do k
        @inbounds if k <= 12
            # Vertical: index k-1, row-major in (3,4) → i=(k-1)÷4, j=(k-1)%4
            i, j = divrem(k - 1, 4)
            (c(cc, i, j) << 4) + c(cc, i + 1, j)
        else
            # Horizontal: index k-13, row-major in (4,3) → i=(k-13)÷3, j=(k-13)%3
            i, j = divrem(k - 13, 3)
            (c(cc, i, j) << 4) + c(cc, i, j + 1)
        end
    end
end

# ============================================================================
# N=3: adjacent triples + L-shapes (52 features, each in 0..4095)
# x_vert: (2,4)=8, x_hor: (4,2)=8, x_ex_00..x_ex_11: (3,3)=9 each → 8+8+36=52
# ============================================================================
function features_3(board::Board)::NTuple{52, Int}
    cc = all_cells(board)
    ntuple(Val(52)) do k
        @inbounds if k <= 8
            # Vertical triples: (i,j),(i+1,j),(i+2,j) ravel (2,4)
            i, j = divrem(k - 1, 4)
            (c(cc, i, j) << 8) + (c(cc, i + 1, j) << 4) + c(cc, i + 2, j)
        elseif k <= 16
            # Horizontal triples: (i,j),(i,j+1),(i,j+2) ravel (4,2)
            i, j = divrem(k - 9, 2)
            (c(cc, i, j) << 8) + (c(cc, i, j + 1) << 4) + c(cc, i, j + 2)
        elseif k <= 25
            # x_ex_00: (x[i+1,j]<<8)+(x[i+1,j+1]<<4)+x[i,j+1], ravel (3,3)
            i, j = divrem(k - 17, 3)
            (c(cc, i + 1, j) << 8) + (c(cc, i + 1, j + 1) << 4) + c(cc, i, j + 1)
        elseif k <= 34
            # x_ex_01: (x[i,j]<<8)+(x[i+1,j]<<4)+x[i+1,j+1], ravel (3,3)
            i, j = divrem(k - 26, 3)
            (c(cc, i, j) << 8) + (c(cc, i + 1, j) << 4) + c(cc, i + 1, j + 1)
        elseif k <= 43
            # x_ex_10: (x[i,j]<<8)+(x[i,j+1]<<4)+x[i+1,j+1], ravel (3,3)
            i, j = divrem(k - 35, 3)
            (c(cc, i, j) << 8) + (c(cc, i, j + 1) << 4) + c(cc, i + 1, j + 1)
        else
            # x_ex_11: (x[i,j]<<8)+(x[i+1,j]<<4)+x[i,j+1], ravel (3,3)
            i, j = divrem(k - 44, 3)
            (c(cc, i, j) << 8) + (c(cc, i + 1, j) << 4) + c(cc, i, j + 1)
        end
    end
end

# ============================================================================
# N=4: columns + rows + 2x2 squares (17 features, each in 0..65535)
# x_vert: 4 columns, x_hor: 4 rows, x_sq: (3,3)=9 squares
# ============================================================================
function features_4(board::Board)::NTuple{17, Int}
    cc = all_cells(board)
    ntuple(Val(17)) do k
        @inbounds if k <= 4
            # Columns: j=k-1, cells (0,j),(1,j),(2,j),(3,j)
            j = k - 1
            (c(cc, 0, j) << 12) + (c(cc, 1, j) << 8) + (c(cc, 2, j) << 4) + c(cc, 3, j)
        elseif k <= 8
            # Rows: i=k-5, cells (i,0),(i,1),(i,2),(i,3)
            i = k - 5
            (c(cc, i, 0) << 12) + (c(cc, i, 1) << 8) + (c(cc, i, 2) << 4) + c(cc, i, 3)
        else
            # 2x2 squares: (i,j),(i+1,j),(i,j+1),(i+1,j+1) ravel (3,3)
            i, j = divrem(k - 9, 3)
            (c(cc, i, j) << 12) + (c(cc, i + 1, j) << 8) + (c(cc, i, j + 1) << 4) + c(cc, i + 1, j + 1)
        end
    end
end

# ============================================================================
# N=5: N=4 features + 4 cross patterns (21 features)
# First 17: same as N=4 (indices 0..65535)
# Last 4: cross at (i,j) for i∈{1,2},j∈{1,2} (indices 0..16^5-1=1048575)
#   center<<16 + above<<12 + left<<8 + below<<4 + right
# ============================================================================
function features_5(board::Board)::NTuple{21, Int}
    cc = all_cells(board)
    ntuple(Val(21)) do k
        @inbounds if k <= 4
            j = k - 1
            (c(cc, 0, j) << 12) + (c(cc, 1, j) << 8) + (c(cc, 2, j) << 4) + c(cc, 3, j)
        elseif k <= 8
            i = k - 5
            (c(cc, i, 0) << 12) + (c(cc, i, 1) << 8) + (c(cc, i, 2) << 4) + c(cc, i, 3)
        elseif k <= 17
            i, j = divrem(k - 9, 3)
            (c(cc, i, j) << 12) + (c(cc, i + 1, j) << 8) + (c(cc, i, j + 1) << 4) + c(cc, i + 1, j + 1)
        else
            # Cross patterns: ravel (2,2) over i∈{1,2}, j∈{1,2}
            i, j = divrem(k - 18, 2)
            i += 1; j += 1
            (c(cc, i, j) << 16) + (c(cc, i - 1, j) << 12) + (c(cc, i, j - 1) << 8) +
            (c(cc, i + 1, j) << 4) + c(cc, i, j + 1)
        end
    end
end

# ============================================================================
# N=6: N=5 features + 12 rectangle features (33 features)
# First 21: same as N=5
# Next 6: vertical 3×2 blocks, ravel (2,3)
# Last 6: horizontal 2×3 blocks, ravel (3,2)
# cutoff controls max cell value for 6-tuples: 13 (original), 14, 15 (=no clamp)
# Base = cutoff + 1. Weight size per 6-tuple group = base^6
# ============================================================================

# Global setting for N=6 cutoff — set before creating agent
# 13 = original Python (base 14, 7.5M per group, 365 MB total)
# 14 = base 15, 11.4M per group, 590 MB total
# 15 = no clamp (base 16 = 16^6, 16.8M per group, 870 MB total)
const N6_CUTOFF = Ref(13)

function set_n6_cutoff!(cutoff::Int)
    cutoff < 13 && error("Cutoff must be >= 13")
    cutoff > 15 && error("Cutoff must be <= 15 (4 bits)")
    N6_CUTOFF[] = cutoff
    base = cutoff + 1
    mb = 12 * base^6 * 8 / 1024 / 1024  # Float64
    println("N=6 cutoff set to $cutoff (base $base, 6-tuple groups: $(base^6) entries, ~$(round(Int, mb)) MB for 12 groups)")
end

# Precomputed multipliers for mixed-radix encoding
@inline function encode_6tuple(cc, r1, c1, r2, c2, r3, c3, r4, c4, r5, c5, r6, c6, cutoff::Int)
    base = cutoff + 1
    b5 = base^5; b4 = base^4; b3 = base^3; b2 = base^2
    b5 * min(c(cc, r1, c1), cutoff) + b4 * min(c(cc, r2, c2), cutoff) +
    b3 * min(c(cc, r3, c3), cutoff) + b2 * min(c(cc, r4, c4), cutoff) +
    base * min(c(cc, r5, c5), cutoff) + min(c(cc, r6, c6), cutoff)
end

function features_6(board::Board)::NTuple{33, Int}
    cc = all_cells(board)
    cutoff = N6_CUTOFF[]
    ntuple(Val(33)) do k
        @inbounds if k <= 4
            j = k - 1
            (c(cc, 0, j) << 12) + (c(cc, 1, j) << 8) + (c(cc, 2, j) << 4) + c(cc, 3, j)
        elseif k <= 8
            i = k - 5
            (c(cc, i, 0) << 12) + (c(cc, i, 1) << 8) + (c(cc, i, 2) << 4) + c(cc, i, 3)
        elseif k <= 17
            i, j = divrem(k - 9, 3)
            (c(cc, i, j) << 12) + (c(cc, i + 1, j) << 8) + (c(cc, i, j + 1) << 4) + c(cc, i + 1, j + 1)
        elseif k <= 21
            i, j = divrem(k - 18, 2)
            i += 1; j += 1
            (c(cc, i, j) << 16) + (c(cc, i - 1, j) << 12) + (c(cc, i, j - 1) << 8) +
            (c(cc, i + 1, j) << 4) + c(cc, i, j + 1)
        elseif k <= 27
            # Vertical 3×2 blocks: ravel (2,3) → i∈{0,1}, j∈{0,1,2}
            i, j = divrem(k - 22, 3)
            encode_6tuple(cc, i,j, i+1,j, i+2,j, i,j+1, i+1,j+1, i+2,j+1, cutoff)
        else
            # Horizontal 2×3 blocks: ravel (3,2) → i∈{0,1,2}, j∈{0,1}
            i, j = divrem(k - 28, 2)
            encode_6tuple(cc, i,j, i,j+1, i,j+2, i+1,j, i+1,j+1, i+1,j+2, cutoff)
        end
    end
end

# ============================================================================
# Feature function dispatch and metadata
# ============================================================================

# Number of features and weight group sizes for each N
# Returns vector of (num_entries_in_group,) for each feature group
function weight_group_sizes(n::Int)::Vector{Int}
    if n == 2
        fill(256, 24)              # 24 groups, each 16^2
    elseif n == 3
        fill(4096, 52)             # 52 groups, each 16^3
    elseif n == 4
        fill(65536, 17)            # 17 groups, each 16^4
    elseif n == 5
        [fill(65536, 17); fill(1048576, 4)]   # 17 × 16^4 + 4 × 16^5
    elseif n == 6
        base6 = N6_CUTOFF[] + 1
        [fill(65536, 17); fill(1048576, 4); fill(base6^6, 12)]
    else
        error("N=$n not yet supported")
    end
end

# Dispatch to the right feature function
function get_feature_function(n::Int)
    if n == 2
        features_2
    elseif n == 3
        features_3
    elseif n == 4
        features_4
    elseif n == 5
        features_5
    elseif n == 6
        features_6
    else
        error("N=$n not yet supported")
    end
end

# Number of feature groups for each N
function num_features(n::Int)::Int
    if n == 2; 24
    elseif n == 3; 52
    elseif n == 4; 17
    elseif n == 5; 21
    elseif n == 6; 33
    else error("N=$n not yet supported")
    end
end

# Base (radix) for each feature group — needed for transfer table construction
function weight_group_bases(n::Int)::Vector{Int}
    if n == 2
        fill(16, 24)
    elseif n == 3
        fill(16, 52)
    elseif n == 4
        fill(16, 17)
    elseif n == 5
        [fill(16, 17); fill(16, 4)]
    elseif n == 6
        [fill(16, 17); fill(16, 4); fill(N6_CUTOFF[] + 1, 12)]
    else
        error("N=$n not yet supported")
    end
end

# Tuple size (number of cells) per feature group
function weight_group_tuple_sizes(n::Int)::Vector{Int}
    if n == 2
        fill(2, 24)
    elseif n == 3
        fill(3, 52)
    elseif n == 4
        fill(4, 17)
    elseif n == 5
        [fill(4, 17); fill(5, 4)]
    elseif n == 6
        [fill(4, 17); fill(5, 4); fill(6, 12)]
    else
        error("N=$n not yet supported")
    end
end
