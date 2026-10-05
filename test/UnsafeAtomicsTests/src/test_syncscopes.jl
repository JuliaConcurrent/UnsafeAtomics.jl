module TestSyncScopes

using UnsafeAtomics: UnsafeAtomics, SyncScope
using Test

using ..Helpers

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

# the primitives pass names to LLVM verbatim, also when they need escaping
odd_load(p) = UnsafeAtomics.Internal.llvm_load(p, Val(:monotonic), Val(Symbol("a\"b\\c")),
                                                Val(false), Val(8), Val(()))

function test_escaping()
    ir = llvm_ir(odd_load, Tuple{Ptr{Int}})
    @test occursin("syncscope(\"a\\22b\\\\c\")", ir)
end

end  # module
