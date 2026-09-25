module TestCore

using UnsafeAtomics: UnsafeAtomics, unordered, monotonic, acquire, release, acq_rel, seq_cst, right
using UnsafeAtomics: none, singlethread
using UnsafeAtomics.Internal: OP_RMW_TABLE, inttypes, floattypes
using InteractiveUtils: code_llvm
using Test
using Base: ConcurrencyViolationError

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
        ((op, rmwop) for (op, rmwop) in OP_RMW_TABLE if op in (+, -, max, min))
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

llvm_ir(f, types) = sprint(io -> code_llvm(io, f, types; debuginfo = :none))

scoped_load(ptr) = UnsafeAtomics.load(ptr, acquire, singlethread)
scoped_store!(ptr, x) = UnsafeAtomics.store!(ptr, x, release, singlethread)
scoped_cas!(ptr, cmp, new) = UnsafeAtomics.cas!(ptr, cmp, new, acq_rel, acquire, singlethread)
scoped_add!(ptr, x) = UnsafeAtomics.add!(ptr, x, acq_rel, singlethread)

function test_syncscope_is_emitted()
    # Values alone can't tell whether the scope made it into the instruction.
    @testset for T in [Int32, UInt64, Float64]
        P = Ptr{T}
        scope = raw"syncscope\(\"singlethread\"\)"
        @test occursin(Regex("load atomic .* $scope acquire"), llvm_ir(scoped_load, Tuple{P}))
        @test occursin(Regex("store atomic .* $scope release"), llvm_ir(scoped_store!, Tuple{P,T}))
        @test occursin(Regex("cmpxchg .* $scope acq_rel acquire"), llvm_ir(scoped_cas!, Tuple{P,T,T}))
        @test occursin(Regex("atomicrmw f?add .* $scope acq_rel"), llvm_ir(scoped_add!, Tuple{P,T}))
    end
end

function test_unsupported_arguments()
    # These used to recurse in the `as_native_uint` fallbacks until the stack overflowed.
    unsupported_scope = UnsafeAtomics.Internal.LLVMSyncScope{:workgroup}()
    @testset for T in [Int32, Float32]
        xs = T[1, 2]
        ptr = pointer(xs, 1)
        GC.@preserve xs begin
            @test_throws ConcurrencyViolationError UnsafeAtomics.load(ptr, release)
            @test_throws ConcurrencyViolationError UnsafeAtomics.load(ptr, acq_rel)
            @test_throws ConcurrencyViolationError UnsafeAtomics.store!(ptr, T(3), acquire)
            @test_throws ConcurrencyViolationError UnsafeAtomics.store!(ptr, T(3), acq_rel)
            @test_throws ConcurrencyViolationError UnsafeAtomics.cas!(
                ptr, T(1), T(3), unordered, monotonic)
            @test_throws ConcurrencyViolationError UnsafeAtomics.cas!(
                ptr, T(1), T(3), seq_cst, release)
            @test_throws ArgumentError UnsafeAtomics.load(ptr, monotonic, unsupported_scope)
            @test_throws ArgumentError UnsafeAtomics.store!(
                ptr, T(3), monotonic, unsupported_scope)
            @test_throws ArgumentError UnsafeAtomics.cas!(
                ptr, T(1), T(3), monotonic, monotonic, unsupported_scope)
            @test xs == T[1, 2]
        end
    end
end

function test_failure_order()
    failure_order = UnsafeAtomics.Internal.failure_order
    @test failure_order(monotonic) === monotonic
    @test failure_order(acquire) === acquire
    @test failure_order(release) === monotonic
    @test failure_order(acq_rel) === acquire
    @test failure_order(seq_cst) === seq_cst
end

cas_acq_rel!(ptr, cmp, new) = UnsafeAtomics.cas!(ptr, cmp, new, acq_rel)

function test_cas_single_ordering()
    # A single ordering used to be taken as the failure ordering as well, which is
    # invalid for release and acq_rel.
    @testset for T in [Int32, Float64], ord in [monotonic, acquire, release, acq_rel, seq_cst]
        xs = T[1, 2]
        ptr = pointer(xs, 1)
        GC.@preserve xs begin
            @test UnsafeAtomics.cas!(ptr, T(1), T(3), ord) === (old = T(1), success = true)
            @test UnsafeAtomics.cas!(ptr, T(1), T(4), ord) === (old = T(3), success = false)
            @test xs[1] === T(3)
        end
    end
    @test occursin(r"cmpxchg .* acq_rel acquire", llvm_ir(cas_acq_rel!, Tuple{Ptr{Int32},Int32,Int32}))
end

scoped_max!(ptr, x) = UnsafeAtomics.max!(ptr, x, acq_rel, singlethread)

function test_cas_loop_fallback()
    # Operations without an atomicrmw instruction used to be a MethodError on Ptr.
    xs = Float32[1, 2]
    ptr = pointer(xs, 1)
    GC.@preserve xs begin
        @test UnsafeAtomics.modify!(ptr, max, 3f0) === (1f0 => 3f0)
        @test UnsafeAtomics.modify!(ptr, min, -0f0, acquire) === (3f0 => -0f0)
        @test UnsafeAtomics.modify!(ptr, max, 0f0, release, singlethread) === (-0f0 => 0f0)
        # Julia's `max` propagates NaN, unlike LLVM's `atomicrmw fmax`.
        @test UnsafeAtomics.modify!(ptr, max, NaN32) === (0f0 => NaN32)
        @test xs[1] === NaN32
        @test UnsafeAtomics.min!(ptr, 1f0) === NaN32
        @test isnan(xs[1])
    end

    ys = Int32[3, 4]
    ptr = pointer(ys, 1)
    GC.@preserve ys begin
        @test UnsafeAtomics.modify!(ptr, *, Int32(2)) === (Int32(3) => Int32(6))
        @test UnsafeAtomics.modify!(ptr, (a, b) -> a ÷ b, Int32(4), monotonic) ===
              (Int32(6) => Int32(1))
        @test ys == Int32[1, 4]
    end

    bs = [false, false]
    ptr = pointer(bs, 1)
    GC.@preserve bs begin
        @test UnsafeAtomics.modify!(ptr, |, true) === (false => true)
        @test UnsafeAtomics.xor!(ptr, true, acq_rel) === true
        @test bs == [false, false]
    end

    @test occursin(r"cmpxchg .* syncscope\(\"singlethread\"\) acq_rel monotonic",
                   llvm_ir(scoped_max!, Tuple{Ptr{Float64},Float64}))
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

function catch_fence(ord)
    try
        UnsafeAtomics.fence(ord, none)
    catch err
        return err
    end
    return nothing
end

function test_fence_unordered_error()
    # Julia's inference thinks the intrinsic throws another type of exception, which Julia
    # 1.11 miscompiled when the error was caught, corrupting memory.
    @test catch_fence(unordered) isa Base.ConcurrencyViolationError
    @test catch_fence(monotonic) === nothing
end

function test_fence_weak_orderings()
    # `fence` requires at least `acquire`. The intrinsic accepts `monotonic` and turns
    # it into a no-op, but rejects `unordered`; preserve both behaviors in the fallback.
    @test UnsafeAtomics.fence(monotonic, none) === nothing
    @test_throws Base.ConcurrencyViolationError UnsafeAtomics.fence(unordered, none)
end

end  # module
