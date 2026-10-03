module TestPublic

using UnsafeAtomics: UnsafeAtomics
using Test

function test_public()
    isdefined(Base, :ispublic) || return
    @testset for name in [:Ordering, :SyncScope,
                          :load, :store!, :cas!, :modify!, :fence,
                          :add!, :sub!, :xchg!, :and!, :nand!, :or!, :xor!, :max!, :min!, :right,
                          :unordered, :monotonic, :acquire, :release, :acq_rel, :seq_cst,
                          :acquire_release, :sequentially_consistent,
                          :none, :singlethread, :subgroup, :workgroup, :device, :system]
        @test Base.ispublic(UnsafeAtomics, name)
        @test !Base.isexported(UnsafeAtomics, name)
    end
    @test !Base.ispublic(UnsafeAtomics, :Internal)
end

end  # module
