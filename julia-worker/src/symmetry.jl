# D4 symmetry group — 8 transformations of a 4x4 board
#
# Python's update loop traces through these boards:
#   x, T(x), rot90(x), T(rot90(x)), rot180(x), T(rot180(x)), rot270(x), T(rot270(x))
# That's 4 rotations + 4 reflections = full D4 group.

function d4_symmetries(board::Board)::NTuple{8, Board}
    r0 = board                        # identity
    r1 = rot90_ccw(r0)                # 90° ccw
    r2 = rot90_ccw(r1)                # 180°
    r3 = rot90_ccw(r2)                # 270° ccw
    t0 = transpose_board(r0)          # transpose
    t1 = transpose_board(r1)          # transpose of 90°
    t2 = transpose_board(r2)          # transpose of 180°
    t3 = transpose_board(r3)          # transpose of 270°
    (r0, t0, r1, t1, r2, t2, r3, t3)
end
