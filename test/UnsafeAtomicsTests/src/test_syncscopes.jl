module TestSyncScopes

using UnsafeAtomics: UnsafeAtomics, SyncScope
using Test

const SYNCSCOPES = [:none, :singlethread, :subgroup, :workgroup, :device]

function check_syncscope(name::Symbol)
    sync = getfield(UnsafeAtomics, name)
    @test sync isa SyncScope
    @test string(sync) == sprint(print, sync) == string(sync)
    @test repr(sync) == "UnsafeAtomics.$name"
end

function test_syncscope()
    @testset for name in SYNCSCOPES
        check_syncscope(name)
    end
end

function test_system()
    @test UnsafeAtomics.system === UnsafeAtomics.none
    @test string(UnsafeAtomics.system) == ""
end

function test_constructor()
    @testset for name in SYNCSCOPES
        @test SyncScope(name) === getfield(UnsafeAtomics, name)
    end
    @test SyncScope(:system) === UnsafeAtomics.system
    agent = SyncScope(:agent)
    @test agent isa SyncScope
    @test agent === SyncScope(:agent)
    @test string(agent) == "syncscope(\"agent\")"
    @test repr(agent) == "UnsafeAtomics.SyncScope(:agent)"
    @test @inferred((() -> SyncScope(:workgroup))()) === UnsafeAtomics.workgroup
    @test string(SyncScope(Symbol("a\"b\\c"))) == "syncscope(\"a\\22b\\5Cc\")"
end

end  # module
