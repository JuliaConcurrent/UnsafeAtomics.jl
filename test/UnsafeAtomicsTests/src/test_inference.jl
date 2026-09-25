module TestInference

using UnsafeAtomics: UnsafeAtomics, unordered, monotonic, acquire, release, acq_rel, seq_cst
using UnsafeAtomics: singlethread, subgroup, workgroup, device, system
using Core: LLVMPtr
using InteractiveUtils: code_llvm
using Test

# Constant orderings and scopes, as objects or as Symbols, give a single instruction.
add_constant(p, x) = UnsafeAtomics.add!(p, x, acquire, device)
add_symbols(p, x) = UnsafeAtomics.add!(p, x, :acquire, :device)
add_alias(p, x) = UnsafeAtomics.add!(p, x, :acquire_release, :workgroup)
add_agent(p, x) = UnsafeAtomics.add!(p, x, monotonic, UnsafeAtomics.SyncScope(:agent))
modify_symbols(p, x) = UnsafeAtomics.modify!(p, +, x, :seq_cst, :subgroup)
load_symbols(p) = UnsafeAtomics.load(p, :sequentially_consistent, :singlethread)
store_symbols(p, x) = UnsafeAtomics.store!(p, x, :release, :system)
cas_symbols(p, c, n) = UnsafeAtomics.cas!(p, c, n, :acq_rel, :acquire, :device)
cas_default_failure(p, c, n) = UnsafeAtomics.cas!(p, c, n, :acq_rel)
fence_symbols() = UnsafeAtomics.fence(:acquire_release, :workgroup)

const P = LLVMPtr{Int32,1}

# without the counters that code coverage adds (`atomicrmw add` on a constant address)
llvm_ir(f, types) =
    join(filter(!contains("inttoptr ("),
                split(sprint(io -> code_llvm(io, f, types; debuginfo = :none)), '\n')), '\n')

instructions(ir, instruction) = [strip(l) for l in split(ir, '\n') if occursin(instruction, l)]

function check(f, types, instruction, text, rettype)
    ir = llvm_ir(f, types)
    @test endswith(only(instructions(ir, instruction)), text)
    @test !occursin(r"apply_generic|jl_invoke|jl_f_|throw", ir)
    @test only(Base.return_types(f, types)) == rettype
end

function test_constant_orderings()
    check(add_constant, Tuple{P,Int32}, "atomicrmw", "syncscope(\"device\") acquire, align 4", Int32)
    check(add_symbols, Tuple{P,Int32}, "atomicrmw", "syncscope(\"device\") acquire, align 4", Int32)
    check(add_alias, Tuple{P,Int32}, "atomicrmw", "syncscope(\"workgroup\") acq_rel, align 4", Int32)
    check(add_agent, Tuple{P,Int32}, "atomicrmw", "syncscope(\"agent\") monotonic, align 4", Int32)
    check(modify_symbols, Tuple{P,Int32}, "atomicrmw", "syncscope(\"subgroup\") seq_cst, align 4",
          Pair{Int32,Int32})
    check(load_symbols, Tuple{P}, "load atomic", "syncscope(\"singlethread\") seq_cst, align 4", Int32)
    check(store_symbols, Tuple{P,Int32}, "store atomic", " release, align 4", Nothing)
    @test !occursin("syncscope", llvm_ir(store_symbols, Tuple{P,Int32}))
    check(cas_symbols, Tuple{P,Int32,Int32}, "cmpxchg", "syncscope(\"device\") acq_rel acquire, align 4",
          @NamedTuple{old::Int32, success::Bool})
    # the failure ordering is derived from a Symbol too
    check(cas_default_failure, Tuple{P,Int32,Int32}, "cmpxchg", " acq_rel acquire, align 4",
          @NamedTuple{old::Int32, success::Bool})
    check(fence_symbols, Tuple{}, r"^\s*fence ", "fence syncscope(\"workgroup\") acq_rel", Nothing)
end

function test_invalid_constants()
    xs = Int32[0]
    GC.@preserve xs begin
        ptr = pointer(xs)
        @test_throws Base.ConcurrencyViolationError UnsafeAtomics.add!(ptr, Int32(1), :unordered)
        @test_throws Base.ConcurrencyViolationError UnsafeAtomics.add!(ptr, Int32(1), :relaxed)
        @test_throws Base.ConcurrencyViolationError UnsafeAtomics.load(ptr, 1)
        @test_throws ArgumentError UnsafeAtomics.add!(ptr, Int32(1), monotonic, :agent)
        @test_throws ArgumentError UnsafeAtomics.load(ptr, monotonic, 1)
        @test xs[1] == 0
    end
    # invalid orderings for the system-scope fence never reach Julia's intrinsic
    @test_throws Base.ConcurrencyViolationError UnsafeAtomics.fence(:bogus)
    @test_throws Base.ConcurrencyViolationError UnsafeAtomics.fence(:unordered)
    @test UnsafeAtomics.fence(:acquire_release) === nothing
    # The error for an invalid scope has a literal message: GPU compilers don't fold `*` on
    # strings, and can't allocate one.
    src = only(code_lowered(UnsafeAtomics.Internal.throw_invalid_scope, Tuple{}))
    @test !any(src.code) do ex
        f = Meta.isexpr(ex, :call) ? ex.args[1] : ex   # Julia 1.12 refers to `*` separately
        f isa GlobalRef && f.name === :*
    end
end

# Values that are only known at run time are a dynamic call, like for Julia's intrinsics:
# slow, but correct on the CPU.
pick_order(i) = i == 1 ? monotonic : i == 2 ? acquire : i == 3 ? seq_cst : i == 4 ? acq_rel : release
pick_scope(i) = i == 1 ? :singlethread : i == 2 ? subgroup : i == 3 ? :workgroup : i == 4 ? device : system

function test_runtime_values()
    xs = Int32[0]
    GC.@preserve xs begin
        ptr = pointer(xs)
        for (i, o) in enumerate((monotonic, acquire, seq_cst, acq_rel, release))
            @test UnsafeAtomics.add!(ptr, Int32(1), pick_order(i)) == 2(i - 1)
            @test UnsafeAtomics.add!(ptr, Int32(1), Symbol(o)) == 2i - 1
        end
        for i in 1:5
            @test UnsafeAtomics.load(ptr, monotonic, pick_scope(i)) == 10
        end
        @test UnsafeAtomics.cas!(ptr, Int32(10), Int32(12), pick_order(4), pick_order(2)) ===
              (old = Int32(10), success = true)
        @test_throws Base.ConcurrencyViolationError UnsafeAtomics.add!(ptr, Int32(1), Symbol("unordered"))
        @test xs[1] == 12
    end
end

end  # module
