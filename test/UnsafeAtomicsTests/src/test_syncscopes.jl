module TestSyncScopes

using UnsafeAtomics: UnsafeAtomics, SyncScope
using InteractiveUtils: code_llvm
using Test

const SYNCSCOPES = [:singlethread, :subgroup, :workgroup, :device, :system]

function check_syncscope(name::Symbol)
    sync = getfield(UnsafeAtomics, name)
    @test sync isa SyncScope
    @test repr(sync) == "UnsafeAtomics.$name"
end

function test_syncscope()
    @testset for name in SYNCSCOPES
        check_syncscope(name)
    end
end

function test_system()
    @test UnsafeAtomics.none === UnsafeAtomics.system
end

function test_constructor()
    @testset for name in SYNCSCOPES
        @test SyncScope(name) === getfield(UnsafeAtomics, name)
    end
    agent = SyncScope(:agent)
    @test agent isa SyncScope
    @test agent === SyncScope(:agent)
    @test repr(agent) == "UnsafeAtomics.SyncScope(:agent)"
    @test @inferred((() -> SyncScope(:workgroup))()) === UnsafeAtomics.workgroup
end

# names are passed to LLVM verbatim, also when they need escaping
const ODD_SCOPE = SyncScope(Symbol("a\"b\\c"))
odd_load(p) = UnsafeAtomics.load(p, UnsafeAtomics.monotonic, ODD_SCOPE)

function test_escaping()
    ir = sprint(io -> code_llvm(io, odd_load, Tuple{Ptr{Int}}; debuginfo = :none))
    @test occursin("syncscope(\"a\\22b\\\\c\")", ir)
end

end  # module
