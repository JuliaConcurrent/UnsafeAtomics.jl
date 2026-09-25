module UnsafeAtomicsTests

include("bits.jl")
include("test_orderings.jl")
include("test_syncscopes.jl")
include("test_core.jl")
include("test_generator.jl")
include("test_llvmptr.jl")
include("test_public.jl")

end  # module UnsafeAtomicsTests
