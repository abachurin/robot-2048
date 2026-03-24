module Game2048

using Random

include("board.jl")
include("lookup.jl")
include("symmetry.jl")
include("features.jl")
include("weights.jl")
include("game.jl")
include("agent.jl")

# Initialize lookup tables at module load time
function __init__()
    generate_tables!()
end

end # module
