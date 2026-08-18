module TestCore

using UnsafeAtomics: UnsafeAtomics, monotonic, acquire, release, acq_rel, seq_cst, right
using UnsafeAtomics: none, singlethread
using UnsafeAtomics.Internal: OP_RMW_TABLE, inttypes, floattypes
using InteractiveUtils: code_llvm
using Test

using ..Bits

function test_default_ordering()
    @testset for T in inttypes
        test_default_ordering(T)
    end
    @testset for T in floattypes
        test_default_ordering(T)
    end
    @testset for T in (asbits(T) for T in inttypes if T <: Unsigned)
        test_default_ordering(T)
    end
    UnsafeAtomics.fence()
end

rmw_table_for(@nospecialize T) =
    if T <: AbstractFloat
        ((op, rmwop) for (op, rmwop) in OP_RMW_TABLE if op in (+, -))
    elseif T <: AbstractBits
        ((op, rmwop) for (op, rmwop) in OP_RMW_TABLE if op in (right,))
    else
        OP_RMW_TABLE
    end

function test_default_ordering(T::Type)
    xs = T[rand(T), rand(T)]
    x1 = rand(T)
    x2 = rand(T)
    @debug "xs=$(repr(xs)) x1=$(repr(x1)) x2=$(repr(x2))"

    ptr = pointer(xs, 1)
    GC.@preserve xs begin
        @test UnsafeAtomics.load(ptr) === xs[1]
        UnsafeAtomics.store!(ptr, x1)
        @test xs[1] === x1
        desired = (old = x1, success = true)
        @test UnsafeAtomics.cas!(ptr, x1, x2) === (old = x1, success = true)
        @test xs[1] === x2
        @testset for (op, name) in rmw_table_for(T)
            xs[1] = x1
            @test UnsafeAtomics.modify!(ptr, op, x2) === (x1 => op(x1, x2))
            @test xs[1] === op(x1, x2)

            rmw = getfield(UnsafeAtomics, Symbol(name, :!))
            xs[1] = x1
            @test rmw(ptr, x2) === x1
            @test xs[1] === op(x1, x2)
        end
    end
end

function test_explicit_ordering()
    @testset for T in [UInt, Float64]
        test_explicit_ordering(T)
    end
    UnsafeAtomics.fence(monotonic)
    UnsafeAtomics.fence(acquire)
    UnsafeAtomics.fence(release)
    UnsafeAtomics.fence(acq_rel)
    UnsafeAtomics.fence(seq_cst)
end

function test_explicit_ordering(T::Type)
    xs = T[rand(T), rand(T)]
    x1 = rand(T)
    x2 = rand(T)
    @debug "xs=$(repr(xs)) x1=$(repr(x1)) x2=$(repr(x2))"

    ptr = pointer(xs, 1)
    GC.@preserve xs begin

        @test UnsafeAtomics.load(ptr, acquire) === xs[1]
        UnsafeAtomics.store!(ptr, x1, release)
        @test xs[1] === x1
        desired = (old = x1, success = true)
        @test UnsafeAtomics.cas!(ptr, x1, x2, acq_rel, acquire) === desired
        @test xs[1] === x2
        @testset for (op, name) in rmw_table_for(T)
            xs[1] = x1
            @test UnsafeAtomics.modify!(ptr, op, x2, acq_rel) === (x1 => op(x1, x2))
            @test xs[1] === op(x1, x2)

            rmw = getfield(UnsafeAtomics, Symbol(name, :!))
            xs[1] = x1
            @test rmw(ptr, x2, acquire) === x1
            @test xs[1] === op(x1, x2)
        end
    end
end

function test_explicit_syncscope(T::Type)
    xs = T[rand(T), rand(T)]
    x1 = rand(T)
    x2 = rand(T)
    @debug "xs=$(repr(xs)) x1=$(repr(x1)) x2=$(repr(x2))"

    ptr = pointer(xs, 1)
    GC.@preserve xs begin

        @test UnsafeAtomics.load(ptr, acquire, none) === xs[1]
        @test UnsafeAtomics.load(ptr, acquire, singlethread) === xs[1]
        UnsafeAtomics.store!(ptr, x1, release, singlethread)
        @test xs[1] === x1
        desired = (old = x1, success = true)
        @test UnsafeAtomics.cas!(ptr, x1, x2, acq_rel, acquire, singlethread) === desired
        @test xs[1] === x2
        @testset for (op, name) in rmw_table_for(T)
            xs[1] = x1
            @test UnsafeAtomics.modify!(ptr, op, x2, acq_rel, singlethread) === (x1 => op(x1, x2))
            @test xs[1] === op(x1, x2)

            rmw = getfield(UnsafeAtomics, Symbol(name, :!))
            xs[1] = x1
            @test rmw(ptr, x2, acquire, singlethread) === x1
            @test xs[1] === op(x1, x2)
        end
    end
end

function test_explicit_syncscope()
    @testset for T in [UInt, Float64]
        test_explicit_syncscope(T)
    end
    UnsafeAtomics.fence(monotonic, none)
    UnsafeAtomics.fence(acquire, singlethread)
    UnsafeAtomics.fence(release, singlethread)
    UnsafeAtomics.fence(acq_rel, none)
    UnsafeAtomics.fence(seq_cst, none)
end

barrier_acquire() = (UnsafeAtomics.fence(acquire); nothing)
barrier_release() = (UnsafeAtomics.fence(release); nothing)
barrier_acq_rel() = (UnsafeAtomics.fence(acq_rel); nothing)
barrier_seq_cst() = (UnsafeAtomics.fence(seq_cst); nothing)

function test_fence_is_emitted()
    # Exercise the public wrapper: a direct intrinsic call survives even on affected
    # Julia versions, while a wrapper call can be deleted when its result is unused.
    # `seq_cst` may lower to inline asm rather than an LLVM `fence` (see core.jl).
    @testset for f in (barrier_acquire, barrier_release, barrier_acq_rel, barrier_seq_cst)
        ir = sprint(io -> code_llvm(io, f, Tuple{}; optimize = true, debuginfo = :none))
        @test occursin(r"^\s*fence "m, ir) || occursin("asm sideeffect", ir)
    end
end

end  # module
