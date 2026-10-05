module TestPublic

using UnsafeAtomics: UnsafeAtomics
using Test

function test_public()
    isdefined(Base, :ispublic) || return
    @testset for name in [:Ordering, :SyncScope,
                          :load, :store!, :cas!, :modify!, :fence,
                          :add!, :sub!, :xchg!, :and!, :nand!, :or!, :xor!, :max!, :min!, :fmax!, :fmin!,
                          :inc_wrap!, :dec_wrap!, :sub_cond!, :sub_sat!,
                          :right, :fmax, :fmin, :inc_wrap, :dec_wrap, :sub_cond, :sub_sat,
                          :unordered, :monotonic, :acquire, :release, :acq_rel, :seq_cst,
                          :acquire_release, :sequentially_consistent,
                          :none, :singlethread, :subgroup, :workgroup, :device, :system,
                          :failure_order]
        @test Base.ispublic(UnsafeAtomics, name)
        @test !Base.isexported(UnsafeAtomics, name)
    end
    @test !Base.ispublic(UnsafeAtomics, :Internal)
    @testset for name in [:llvm_load, :llvm_store!, :llvm_rmw!, :llvm_modify!, :llvm_cmpxchg!,
                          :llvm_fence]
        @test Base.ispublic(UnsafeAtomics.Internal, name)
    end
end

end  # module
