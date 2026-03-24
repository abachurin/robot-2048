# Weight storage: single contiguous vector with offset indexing
# Parameterized by float type T (Float16, Float32, Float64)
# Optional: "seen" bitvector for lazy weight initialization via transfer

struct WeightTable{T <: AbstractFloat}
    data::Vector{T}
    offsets::Vector{Int}     # 1-based offset into data for each group
    sizes::Vector{Int}       # number of entries per group
    bases::Vector{Int}       # radix per group (16 for base-16, 14 for capped 6-tuples, etc.)
    tuple_sizes::Vector{Int} # number of cells per group
    num_groups::Int
    seen::Union{BitVector, Nothing}  # 1 bit per weight: has this weight been updated?
end

function WeightTable{T}(group_sizes::Vector{Int};
                        init_scale::Float64 = 0.01,
                        enable_transfer::Bool = false,
                        group_bases::Union{Vector{Int}, Nothing} = nothing,
                        group_tuple_sizes::Union{Vector{Int}, Nothing} = nothing,
                        ) where T
    total = sum(group_sizes)
    data = T.(rand(Float64, total) .* init_scale)
    offsets = Vector{Int}(undef, length(group_sizes))
    offset = 1
    for (i, s) in enumerate(group_sizes)
        offsets[i] = offset
        offset += s
    end

    bases = group_bases !== nothing ? group_bases : fill(16, length(group_sizes))
    tsizes = group_tuple_sizes !== nothing ? group_tuple_sizes : fill(4, length(group_sizes))
    seen = enable_transfer ? falses(total) : nothing

    WeightTable{T}(data, offsets, group_sizes, bases, tsizes, length(group_sizes), seen)
end

# Default to Float32
WeightTable(group_sizes::Vector{Int}; kwargs...) = WeightTable{Float32}(group_sizes; kwargs...)

# Total memory used by weights in bytes
memory_bytes(wt::WeightTable) = sizeof(wt.data)
memory_mb(wt::WeightTable) = memory_bytes(wt) / (1024 * 1024)

function transfer_overhead_mb(wt::WeightTable)
    wt.seen !== nothing ? sizeof(wt.seen) / 1024 / 1024 : 0.0
end

# Compute "one level lower" index: decode feature index into cells,
# find first cell with max value, reduce by 1, re-encode.
# Returns 0 if all cells are 0 (no source to transfer from).
@inline function lower_index(idx::Int, base::Int, tsize::Int)::Int
    # Decode into fixed-size buffer (max 8 cells, stack allocated)
    c1 = c2 = c3 = c4 = c5 = c6 = c7 = c8 = 0
    rem = idx
    tsize >= 8 && (c8 = rem % base; rem ÷= base)
    tsize >= 7 && (c7 = rem % base; rem ÷= base)
    tsize >= 6 && (c6 = rem % base; rem ÷= base)
    tsize >= 5 && (c5 = rem % base; rem ÷= base)
    tsize >= 4 && (c4 = rem % base; rem ÷= base)
    tsize >= 3 && (c3 = rem % base; rem ÷= base)
    tsize >= 2 && (c2 = rem % base; rem ÷= base)
    c1 = rem % base

    maxval = max(c1, c2, c3, c4, c5, c6, c7, c8)
    maxval == 0 && return 0

    # Reduce first cell with maxval by 1
    if     c1 == maxval; c1 -= 1
    elseif c2 == maxval; c2 -= 1
    elseif c3 == maxval; c3 -= 1
    elseif c4 == maxval; c4 -= 1
    elseif c5 == maxval; c5 -= 1
    elseif c6 == maxval; c6 -= 1
    elseif c7 == maxval; c7 -= 1
    else                 c8 -= 1
    end

    # Re-encode
    result = c1
    tsize >= 2 && (result = result * base + c2)
    tsize >= 3 && (result = result * base + c3)
    tsize >= 4 && (result = result * base + c4)
    tsize >= 5 && (result = result * base + c5)
    tsize >= 6 && (result = result * base + c6)
    tsize >= 7 && (result = result * base + c7)
    tsize >= 8 && (result = result * base + c8)
    result
end

# Evaluate: sum weights at feature indices (always accumulate in Float64 for stability)
@inline function evaluate(wt::WeightTable{T}, features::NTuple{N, Int})::Float64 where {T, N}
    s = 0.0
    for g in 1:N
        @inbounds s += Float64(wt.data[wt.offsets[g] + features[g]])
    end
    s
end

# Update: add delta to weights at feature indices
# If transfer is enabled, lazily initialize unseen weights from "one level lower"
@inline function update_weights!(wt::WeightTable{T}, features::NTuple{N, Int}, delta::Float64) where {T, N}
    if wt.seen === nothing
        # No transfer — fast path
        for g in 1:N
            @inbounds wt.data[wt.offsets[g] + features[g]] += T(delta)
        end
    else
        for g in 1:N
            pos = wt.offsets[g] + features[g]
            @inbounds if !wt.seen[pos]
                wt.seen[pos] = true
                src_idx = lower_index(features[g], wt.bases[g], wt.tuple_sizes[g])
                if src_idx > 0
                    wt.data[pos] = wt.data[wt.offsets[g] + src_idx]
                end
            end
            @inbounds wt.data[pos] += T(delta)
        end
    end
end

# Weight signature for serialization (matching Python's weightSignature)
function weight_signature(n::Int)::Vector{Int}
    if n == 6
        [17, 4, 12]
    elseif n == 5
        [17, 4]
    else
        [num_features(n)]
    end
end
