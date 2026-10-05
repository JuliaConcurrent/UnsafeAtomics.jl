module TestCore

using UnsafeAtomics: UnsafeAtomics, unordered, monotonic, acquire, release, acq_rel, seq_cst, right
using UnsafeAtomics: none, singlethread, subgroup, workgroup, device, system, SyncScope
using UnsafeAtomics.Internal: OP_RMW_TABLE
using Core: LLVMPtr
using Test
using Base: ConcurrencyViolationError

using ..Helpers

rmw_table_for(@nospecialize T) =
    if T <: AbstractFloat
        ((op, rmwop) for (op, rmwop) in OP_RMW_TABLE
         if op in (+, -, max, min, UnsafeAtomics.fmax, UnsafeAtomics.fmin))
    elseif T <: AbstractBits
        ((op, rmwop) for (op, rmwop) in OP_RMW_TABLE if op in (right,))
    elseif T <: Unsigned
        ((op, rmwop) for (op, rmwop) in OP_RMW_TABLE
         if !(op in (UnsafeAtomics.fmax, UnsafeAtomics.fmin)))
    else
        ((op, rmwop) for (op, rmwop) in OP_RMW_TABLE if !(op in FLOAT_OPS || op in UNSIGNED_OPS))
    end

const FLOAT_OPS = (UnsafeAtomics.fmax, UnsafeAtomics.fmin)
const UNSIGNED_OPS = (UnsafeAtomics.inc_wrap, UnsafeAtomics.dec_wrap, UnsafeAtomics.sub_cond,
                      UnsafeAtomics.sub_sat)

# Every operation, with the given orderings and scope after the operands.
function check_operations(T, P, load = (), store = (), cas = (), modify = (), rmw = ())
    xs = T[rand(T), rand(T)]
    x1 = rand(T)
    x2 = rand(T)
    @debug "xs=$(repr(xs)) x1=$(repr(x1)) x2=$(repr(x2))"

    ptr = pointer_to(P, xs)
    GC.@preserve xs begin
        @test UnsafeAtomics.load(ptr, load...) === xs[1]
        UnsafeAtomics.store!(ptr, x1, store...)
        @test xs[1] === x1
        @test UnsafeAtomics.cas!(ptr, x1, x2, cas...) === (old = x1, success = true)
        @test xs[1] === x2
        if !(T <: AbstractBits)  # a compare-and-swap loop
            xs[1] = x1
            @test UnsafeAtomics.modify!(ptr, *, x2, modify...) === (x1 => x1 * x2)
        end
        @testset for (op, name) in rmw_table_for(T)
            xs[1] = x1
            @test UnsafeAtomics.modify!(ptr, op, x2, modify...) === (x1 => op(x1, x2))
            @test xs[1] === op(x1, x2)

            rmw! = getfield(UnsafeAtomics, Symbol(name, :!))
            xs[1] = x1
            @test rmw!(ptr, x2, rmw...) === x1
            @test xs[1] === op(x1, x2)
        end
    end
    # an atomicrmw instruction, not a compare-and-swap loop
    T <: Integer && @test occursin("atomicrmw add",
                                   llvm_ir(UnsafeAtomics.modify!, Tuple{typeof(ptr),typeof(+),T,typeof.(modify)...}))
end

function test_default_ordering()
    @testset for T in (inttypes..., floattypes..., (asbits(T) for T in inttypes if T <: Unsigned)...),
                 P in POINTER_KINDS
        check_operations(T, P)
    end
    UnsafeAtomics.fence()
end

function test_explicit_ordering()
    @testset for T in [UInt, Float64], P in POINTER_KINDS
        check_operations(T, P, (acquire,), (release,), (acq_rel, acquire), (acq_rel,), (acquire,))
    end
end

const SCOPES = [singlethread, subgroup, workgroup, device, system, SyncScope(:agent)]

function test_explicit_syncscope()
    @testset for T in [UInt, Float64], P in POINTER_KINDS, scope in SCOPES
        check_operations(T, P, (acquire, scope), (release, scope), (acq_rel, acquire, scope),
                         (acq_rel, scope), (acquire, scope))
    end
end

scoped_load(ptr, scope) = UnsafeAtomics.load(ptr, acquire, scope)
scoped_store!(ptr, x, scope) = UnsafeAtomics.store!(ptr, x, release, scope)
scoped_cas!(ptr, cmp, new, scope) = UnsafeAtomics.cas!(ptr, cmp, new, acq_rel, acquire, scope)
scoped_add!(ptr, x, scope) = UnsafeAtomics.add!(ptr, x, acq_rel, scope)
scoped_fence(ord, scope) = (UnsafeAtomics.fence(ord, scope); nothing)

# The line of `ir` with the atomic instruction, which must mention the right scope.
function scoped_instruction(ir, instruction, scope)
    line = only(filter(contains(instruction), split(ir, '\n')))
    if scope === system
        return !occursin("syncscope", line)
    else
        return occursin("syncscope(\"$(UnsafeAtomics.Internal.llvm_syncscope(scope))\")", line)
    end
end

function test_syncscope_is_emitted()
    # Values alone can't tell whether the scope made it into the instruction.
    @testset for T in [Int32, UInt64, Float64], P in (Ptr{T}, LLVMPtr{T,1}), scope in SCOPES
        S = typeof(scope)
        @test scoped_instruction(llvm_ir(scoped_load, Tuple{P,S}), r"load atomic .* acquire", scope)
        @test scoped_instruction(llvm_ir(scoped_store!, Tuple{P,T,S}), r"store atomic .* release", scope)
        @test scoped_instruction(llvm_ir(scoped_cas!, Tuple{P,T,T,S}), r"cmpxchg .* acq_rel acquire", scope)
        @test scoped_instruction(llvm_ir(scoped_add!, Tuple{P,T,S}), r"atomicrmw f?add .* acq_rel", scope)
    end
    # system-scope fences are tested by test_fence_is_emitted
    @testset for scope in filter(!=(system), SCOPES), ord in [acquire, release, acq_rel, seq_cst]
        ir = llvm_ir(scoped_fence, Tuple{typeof(ord),typeof(scope)})
        name = UnsafeAtomics.Internal.llvm_syncscope(scope)
        @test occursin("fence syncscope(\"$name\") $ord", ir)
    end
end

function test_invalid_arguments()
    # These used to recurse in the `as_native_uint` fallbacks until the stack overflowed, and
    # an unordered atomicrmw used to fail to parse.
    @testset for T in [Int32, Float32], P in POINTER_KINDS
        xs = T[1, 2]
        ptr = pointer_to(P, xs)
        GC.@preserve xs begin
            @test_throws ConcurrencyViolationError UnsafeAtomics.load(ptr, release)
            @test_throws ConcurrencyViolationError UnsafeAtomics.load(ptr, :acquire_release)
            @test_throws ConcurrencyViolationError UnsafeAtomics.store!(ptr, T(3), acquire)
            @test_throws ConcurrencyViolationError UnsafeAtomics.cas!(ptr, T(1), T(3), unordered, monotonic)
            @test_throws ConcurrencyViolationError UnsafeAtomics.cas!(ptr, T(1), T(3), seq_cst, release)
            @test_throws ConcurrencyViolationError UnsafeAtomics.add!(ptr, T(1), unordered)
            @test_throws ConcurrencyViolationError UnsafeAtomics.max!(ptr, T(1), unordered)
            @test_throws ConcurrencyViolationError UnsafeAtomics.modify!(ptr, *, T(1), unordered)
            @test_throws ConcurrencyViolationError UnsafeAtomics.load(ptr, :bogus)
            @test_throws ConcurrencyViolationError UnsafeAtomics.add!(ptr, T(1), :relaxed)
            @test_throws ConcurrencyViolationError UnsafeAtomics.load(ptr, 1)
            # only canonical scopes can be passed as a Symbol
            @test_throws ArgumentError UnsafeAtomics.load(ptr, monotonic, :agent)
            @test_throws ArgumentError UnsafeAtomics.store!(ptr, T(3), monotonic, :agent)
            @test_throws ArgumentError UnsafeAtomics.cas!(ptr, T(1), T(3), monotonic, monotonic, :agent)
            @test_throws ArgumentError UnsafeAtomics.add!(ptr, T(1), monotonic, :agent)
            @test_throws ArgumentError UnsafeAtomics.load(ptr, monotonic, 1)
            @test xs == T[1, 2]
        end
    end
    @test_throws ArgumentError UnsafeAtomics.fence(acquire, :agent)
    # invalid orderings for the system-scope fence never reach Julia's intrinsic
    @test_throws ConcurrencyViolationError UnsafeAtomics.fence(:bogus)
    @test UnsafeAtomics.fence(:acquire_release) === nothing
    # The error for an invalid scope has a literal message: GPU compilers don't fold `*` on
    # strings, and can't allocate one.
    src = only(code_lowered(UnsafeAtomics.Internal.throw_invalid_scope, Tuple{}))
    @test !any(src.code) do ex
        f = Meta.isexpr(ex, :call) ? ex.args[1] : ex   # Julia 1.12 refers to `*` separately
        f isa GlobalRef && f.name === :*
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
    @testset for T in [Int32, Float64], P in POINTER_KINDS, ord in [monotonic, acquire, release, acq_rel, seq_cst]
        xs = T[1, 2]
        ptr = pointer_to(P, xs)
        GC.@preserve xs begin
            @test UnsafeAtomics.cas!(ptr, T(1), T(3), ord) === (old = T(1), success = true)
            @test UnsafeAtomics.cas!(ptr, T(1), T(4), ord) === (old = T(3), success = false)
            @test xs[1] === T(3)
        end
    end
    @test occursin(r"cmpxchg .* acq_rel acquire", llvm_ir(cas_acq_rel!, Tuple{Ptr{Int32},Int32,Int32}))
end

scoped_mul!(ptr, x) = UnsafeAtomics.modify!(ptr, *, x, acq_rel, singlethread)

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
                   llvm_ir(scoped_mul!, Tuple{Ptr{Float64},Float64}))
end

float_modify!(ptr, op, x) = UnsafeAtomics.modify!(ptr, op, x, monotonic, device)

function test_float_minmax()
    # `max` and `min` have Julia's semantics: NaN propagates, and -0.0 < 0.0
    @testset for T in [Float16, Float32, Float64], P in POINTER_KINDS
        xs = T[1, 0]
        ptr = pointer_to(P, xs)
        GC.@preserve xs begin
            @test UnsafeAtomics.max!(ptr, T(-0.0)) === T(1)
            @test UnsafeAtomics.modify!(ptr, max, T(NaN)) === (T(1) => T(NaN))
            @test isnan(xs[1])
            xs[1] = -0.0
            @test UnsafeAtomics.modify!(ptr, max, T(0.0)) === (T(-0.0) => T(0.0))
            @test xs[1] === T(0.0)
            @test UnsafeAtomics.modify!(ptr, min, T(-0.0)) === (T(0.0) => T(-0.0))
            @test xs[1] === T(-0.0)
            @test UnsafeAtomics.min!(ptr, T(NaN)) === T(-0.0)
            @test isnan(xs[1])
        end
    end
    # `fmax` and `fmin` ignore NaN
    @testset for T in [Float16, Float32, Float64], P in POINTER_KINDS
        xs = T[1, 0]
        ptr = pointer_to(P, xs)
        GC.@preserve xs begin
            @test UnsafeAtomics.fmax!(ptr, T(NaN)) === T(1)
            @test xs[1] === T(1)
            @test UnsafeAtomics.modify!(ptr, UnsafeAtomics.fmax, T(2)) === (T(1) => T(2))
            @test UnsafeAtomics.fmin!(ptr, T(NaN), acquire) === T(2)
            @test UnsafeAtomics.modify!(ptr, UnsafeAtomics.fmin, T(-1), release) === (T(2) => T(-1))
            @test xs[1] === T(-1)
        end
    end
    @test UnsafeAtomics.fmax(NaN, 1.0) === 1.0
    @test UnsafeAtomics.fmax(1.0, NaN) === 1.0
    @test UnsafeAtomics.fmin(NaN32, 2f0) === 2f0
    @test isnan(UnsafeAtomics.fmin(NaN, NaN))

    # native where LLVM has the instruction; a compare-and-swap loop otherwise
    P = LLVMPtr{Float32,1}
    ir(op) = llvm_ir(float_modify!, Tuple{P,typeof(op),Float32})
    @test occursin("atomicrmw fmax", ir(UnsafeAtomics.fmax))
    @test occursin("atomicrmw fmin", ir(UnsafeAtomics.fmin))
    if Base.libllvm_version >= v"21"
        @test occursin("atomicrmw fmaximum", ir(max))
        @test occursin("atomicrmw fminimum", ir(min))
    else
        @test occursin("cmpxchg", ir(max)) && !occursin("atomicrmw", ir(max))
        @test occursin("cmpxchg", ir(min)) && !occursin("atomicrmw", ir(min))
    end
end

wrap_modify!(ptr, op, x) = UnsafeAtomics.modify!(ptr, op, x, monotonic, device)

function test_unsigned_ops()
    UA = UnsafeAtomics
    # the semantics of the LLVM instructions
    @test UA.inc_wrap(0x03, 0x05) === 0x04
    @test UA.inc_wrap(0x05, 0x05) === 0x00
    @test UA.inc_wrap(0x07, 0x05) === 0x00
    @test UA.dec_wrap(0x03, 0x05) === 0x02
    @test UA.dec_wrap(0x00, 0x05) === 0x05
    @test UA.dec_wrap(0x07, 0x05) === 0x05
    @test UA.sub_cond(0x07, 0x05) === 0x02
    @test UA.sub_cond(0x03, 0x05) === 0x03
    @test UA.sub_sat(0x07, 0x05) === 0x02
    @test UA.sub_sat(0x03, 0x05) === 0x00
    @test_throws MethodError UA.inc_wrap(1, 2)

    @testset for T in [UInt8, UInt32, UInt64], P in POINTER_KINDS
        xs = T[3, 0]
        ptr = pointer_to(P, xs)
        GC.@preserve xs begin
            @test UA.inc_wrap!(ptr, T(4)) === T(3)
            @test UA.inc_wrap!(ptr, T(4), acquire) === T(4)
            @test UA.dec_wrap!(ptr, T(4), release, device) === T(0)
            @test UA.modify!(ptr, UA.dec_wrap, T(4)) === (T(4) => T(3))
            @test UA.sub_cond!(ptr, T(5)) === T(3)
            @test UA.modify!(ptr, UA.sub_cond, T(2)) === (T(3) => T(1))
            @test UA.sub_sat!(ptr, T(2)) === T(1)
            @test UA.modify!(ptr, UA.sub_sat, T(1), seq_cst, singlethread) === (T(0) => T(0))
            @test xs == T[0, 0]
        end
    end

    # native from the LLVM version that has the instruction; a compare-and-swap loop before
    P = LLVMPtr{UInt32,1}
    for (op, rmw, version) in ((UA.inc_wrap, "uinc_wrap", v"22"), (UA.dec_wrap, "udec_wrap", v"22"),
                               (UA.sub_cond, "usub_cond", v"22"), (UA.sub_sat, "usub_sat", v"22"))
        ir = llvm_ir(wrap_modify!, Tuple{P,typeof(op),UInt32})
        if Base.libllvm_version >= version
            @test occursin("atomicrmw $rmw", ir)
        else
            @test occursin("cmpxchg", ir) && !occursin("atomicrmw", ir)
        end
    end
end

function test_native_rmw()
    # which operations have an atomicrmw instruction, on this version of LLVM
    UA = UnsafeAtomics
    native_rmw = UA.Internal.native_rmw
    llvm = Base.libllvm_version
    bf16 = isdefined(Core, :BFloat16) ? (Core.BFloat16,) : ()
    for T in (Int8, Int32, Int64), (op, rmw) in ((+, :add), (-, :sub), (&, :and), (|, :or),
                                                 (xor, :xor), (⊼, :nand), (max, :max),
                                                 (min, :min), (right, :xchg))
        @test native_rmw(op, T) === rmw
    end
    for T in (UInt8, UInt64)
        @test native_rmw(max, T) === :umax
        @test native_rmw(min, T) === :umin
        @test native_rmw(UA.inc_wrap, T) === (llvm >= v"22" ? :uinc_wrap : nothing)
        @test native_rmw(UA.dec_wrap, T) === (llvm >= v"22" ? :udec_wrap : nothing)
        @test native_rmw(UA.sub_cond, T) === (llvm >= v"22" ? :usub_cond : nothing)
        @test native_rmw(UA.sub_sat, T) === (llvm >= v"22" ? :usub_sat : nothing)
    end
    @test native_rmw(UA.inc_wrap, Int32) === nothing
    for T in bf16
        # the AArch64 back-end can't compile these before LLVM 20
        @test native_rmw(+, T) === (llvm >= v"20" ? :fadd : nothing)
        @test native_rmw(UA.fmax, T) === (llvm >= v"20" ? :fmax : nothing)
        @test native_rmw(right, T) === :xchg
    end
    for T in (Float16, Float32, Float64)
        @test native_rmw(+, T) === :fadd
        @test native_rmw(-, T) === :fsub
        @test native_rmw(UA.fmax, T) === :fmax
        @test native_rmw(UA.fmin, T) === :fmin
        @test native_rmw(max, T) === (llvm >= v"21" ? :fmaximum : nothing)
        @test native_rmw(min, T) === (llvm >= v"21" ? :fminimum : nothing)
        @test native_rmw(right, T) === :xchg
        @test native_rmw(&, T) === nothing
    end
    @test native_rmw(|, Bool) === :or
    @test native_rmw(max, Bool) === :umax
    @test native_rmw(⊼, Bool) === nothing  # a bitwise nand of Bools isn't a Bool
    @test native_rmw(+, Bool) === nothing
    @test native_rmw(right, Ptr{Cvoid}) === :xchg
    @test native_rmw(+, Ptr{Cvoid}) === nothing
    @test native_rmw(right, LLVMPtr{Cvoid,1}) === :xchg
    @test native_rmw(right, Nothing) === :xchg
    @test native_rmw(*, Int32) === nothing
    @test native_rmw((a, b) -> a + b, Int32) === nothing
end

function test_contention()
    # in another process, as this one may only have one thread
    code = """
    using UnsafeAtomics
    const UA = UnsafeAtomics
    ints = zeros(Int, 1); floats = zeros(Float64, 1); small = zeros(Int16, 1); bools = [false]
    n = 2_000 * Threads.nthreads()
    GC.@preserve ints floats small bools begin
        Threads.@threads for i in 1:n
            UA.add!(pointer(ints), 1)                                       # atomicrmw
            UA.modify!(pointer(floats), (a, b) -> a + b, 1.0, UA.acq_rel)   # CAS loop
            UA.max!(pointer(small), Int16(i % 1000), UA.monotonic)
            UA.xor!(pointer(bools), true, UA.monotonic, :workgroup)
            UA.fence(:seq_cst)
        end
    end
    print(Threads.nthreads(), " ", ints[1] == n, " ", floats[1] == n, " ", small[1] == 999, " ",
          bools[1] == isodd(n))
    """
    cmd = `$(Base.julia_cmd()) --startup-file=no --threads=4 --project=$(Base.active_project()) -e $code`
    @test readchomp(addenv(cmd, "JULIA_LOAD_PATH" => join(LOAD_PATH, Sys.iswindows() ? ';' : ':'))) ==
          "4 true true true true"
end

function test_zero_size_values()
    xs = [nothing, nothing]
    GC.@preserve xs for P in POINTER_KINDS
        ptr = pointer_to(P, xs)
        @test UnsafeAtomics.load(ptr) === UnsafeAtomics.store!(ptr, nothing) === nothing
        @test UnsafeAtomics.load(ptr, acquire) === nothing
        @test UnsafeAtomics.store!(ptr, nothing, release, workgroup) === nothing
        @test UnsafeAtomics.xchg!(ptr, nothing) === nothing
        @test UnsafeAtomics.modify!(ptr, right, nothing) === (nothing => nothing)
        @test UnsafeAtomics.cas!(ptr, nothing, nothing) === (old = nothing, success = true)
        @test_throws ConcurrencyViolationError UnsafeAtomics.load(ptr, release)
    end
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
        ir = llvm_ir(f, Tuple{})
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

function test_scoped_fences()
    # `fence` requires at least `acquire`. The intrinsic accepts `monotonic` and turns it into
    # a no-op, but rejects `unordered`; the other scopes behave the same.
    @testset for scope in SCOPES
        for ord in [monotonic, acquire, release, acq_rel, seq_cst]
            @test UnsafeAtomics.fence(ord, scope) === nothing
        end
        @test_throws ConcurrencyViolationError UnsafeAtomics.fence(unordered, scope)
    end
    # the name is escaped in the IR
    @test UnsafeAtomics.fence(acquire, SyncScope(Symbol("a\"b\\c"))) === nothing
end

# from memory: before LLVM 17, the AArch64 back-end can't materialize a bfloat constant
const BFLOAT16_BITS = [0x3f80, 0x4000]  # 1.0, 2.0

function test_bfloat16()
    isdefined(Core, :BFloat16) || return
    one, two = reinterpret(Core.BFloat16, BFLOAT16_BITS)
    bits(x) = reinterpret(UInt16, x)
    xs = [one]
    GC.@preserve xs for P in POINTER_KINDS
        xs[1] = one
        ptr = pointer_to(P, xs)
        @test bits(UnsafeAtomics.xchg!(ptr, two)) === 0x3f80
        # Core.BFloat16 has no arithmetic without BFloat16s.jl, but neither the instruction
        # nor the compare-and-swap loop needs it
        @test bits(UnsafeAtomics.add!(ptr, one)) === 0x4000
        @test bits(xs[1]) === 0x4040  # 3.0
        @test bits(UnsafeAtomics.max!(ptr, two)) === 0x4040
        @test bits(UnsafeAtomics.modify!(ptr, -, one).second) === 0x4000  # 2.0
        @test bits(UnsafeAtomics.modify!(ptr, +, one, UnsafeAtomics.acq_rel).second) === 0x4040
    end
end

# The LLVMPtr methods used to live in a package extension, which was only loaded with LLVM.jl.
# UnsafeAtomics now depends on LLVM.jl, and defines them itself.
function test_without_extension()
    @test Base.get_extension(UnsafeAtomics, :UnsafeAtomicsLLVM) === nothing
    @test Base.return_types(UnsafeAtomics.add!, (Core.LLVMPtr{Int32,1}, Int32)) == [Int32]
end

end  # module
