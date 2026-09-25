module TestInference

using UnsafeAtomics: UnsafeAtomics, unordered, monotonic, acquire, release, acq_rel, seq_cst
using UnsafeAtomics: singlethread, subgroup, workgroup, device, system
using Core: LLVMPtr
using InteractiveUtils: code_llvm
using Test

# Orderings and scopes that are only known at run time, as a Union of more types than Julia
# splits (4). They used to be dispatched on, which gave dynamic calls.
pick_order(i) = i == 1 ? monotonic : i == 2 ? acquire : i == 3 ? seq_cst : i == 4 ? acq_rel : release
# Julia widens a Union of more than 3 scopes to the abstract type, which can also hold other
# scopes, like `SyncScope(:agent)`. Scopes chosen at run time among more are passed as Symbols.
pick_scope(i) = i == 1 ? workgroup : i == 2 ? device : system
pick_scope_symbol(i) = i == 1 ? :singlethread : i == 2 ? :subgroup : i == 3 ? :workgroup : i == 4 ? :device : :system

add_order(p, x, i) = UnsafeAtomics.add!(p, x, pick_order(i), device)
add_symbol(p, x, o::Symbol) = UnsafeAtomics.add!(p, x, o, device)
modify_order(p, x, i) = UnsafeAtomics.modify!(p, +, x, pick_order(i), device)
load_order(p, i) = UnsafeAtomics.load(p, pick_order(i), device)
load_symbol(p, o::Symbol) = UnsafeAtomics.load(p, o, device)
store_order(p, x, i) = UnsafeAtomics.store!(p, x, pick_order(i), device)
cas_orders(p, c, n, i, j) = UnsafeAtomics.cas!(p, c, n, pick_order(i), pick_order(j), device)
cas_symbols(p, c, n, s::Symbol, f::Symbol) = UnsafeAtomics.cas!(p, c, n, s, f, device)
fence_order(i) = UnsafeAtomics.fence(pick_order(i), workgroup)
fence_symbol(o::Symbol) = UnsafeAtomics.fence(o, workgroup)
add_scope(p, x, i) = UnsafeAtomics.add!(p, x, monotonic, pick_scope(i))
add_scope_symbol(p, x, i) = UnsafeAtomics.add!(p, x, monotonic, pick_scope_symbol(i))
add_constant(p, x) = UnsafeAtomics.add!(p, x, acquire, device)
add_constant_symbols(p, x) = UnsafeAtomics.add!(p, x, :acquire, :device)

const P = LLVMPtr{Int32,1}

# One instruction per valid ordering. Before Julia 1.12, comparisons don't narrow a `Union`,
# so branches for orderings that can't occur (e.g. `unordered`) may remain.
# without the counters that code coverage adds (`atomicrmw add` on a constant address)
llvm_ir(f, types) =
    join(filter(!contains("inttoptr ("),
                split(sprint(io -> code_llvm(io, f, types; debuginfo = :none)), '\n')), '\n')

function check(f, types, instruction, n, rettype; max = n)
    ir = llvm_ir(f, types)
    @test n <= count(line -> occursin(instruction, line), split(ir, '\n')) <= max
    @test !occursin(r"apply_generic|jl_invoke|jl_f_", ir)
    @test only(Base.return_types(f, types)) == rettype
end

function test_runtime_orderings()
    check(add_order, Tuple{P,Int32,Int}, "atomicrmw add", 5, Int32; max = 6)
    check(add_symbol, Tuple{P,Int32,Symbol}, "atomicrmw add", 5, Int32)
    check(modify_order, Tuple{P,Int32,Int}, "atomicrmw add", 5, Pair{Int32,Int32}; max = 6)
    # only monotonic, acquire and seq_cst are valid for loads, and stores release instead
    check(load_order, Tuple{P,Int}, "load atomic i32", 3, Int32; max = 4)
    check(load_symbol, Tuple{P,Symbol}, "load atomic i32", 4, Int32)
    check(store_order, Tuple{P,Int32,Int}, "store atomic i32", 3, Nothing; max = 4)
    # the failure ordering can't release
    check(cas_orders, Tuple{P,Int32,Int32,Int,Int}, "cmpxchg", 5 * 3, @NamedTuple{old::Int32, success::Bool}; max = 6 * 6)
    check(cas_symbols, Tuple{P,Int32,Int32,Symbol,Symbol}, "cmpxchg", 5 * 3, @NamedTuple{old::Int32, success::Bool}; max = 6 * 6)
    # monotonic is a no-op
    check(fence_order, Tuple{Int}, r"^\s*fence ", 4, Nothing; max = 5)
    check(fence_symbol, Tuple{Symbol}, r"^\s*fence ", 4, Nothing)
end

function test_runtime_scopes()
    check(add_scope, Tuple{P,Int32,Int}, "atomicrmw add", 3, Int32)
    check(add_scope_symbol, Tuple{P,Int32,Int}, "atomicrmw add", 5, Int32)
end

function test_constant_orderings()
    for f in (add_constant, add_constant_symbols)
        ir = llvm_ir(f, Tuple{P,Int32})
        @test count(line -> occursin("atomicrmw", line), split(ir, '\n')) == 1
        @test occursin("syncscope(\"device\") acquire", ir)
        @test !occursin("throw", ir) && !occursin("call ", ir)
    end
end

function test_runtime_values()
    xs = Int32[0]
    GC.@preserve xs begin
        ptr = pointer(xs)
        for (i, o) in enumerate((monotonic, acquire, seq_cst, acq_rel, release))
            @test @inferred(UnsafeAtomics.add!(ptr, Int32(1), pick_order(i))) == 2(i - 1)
            @test UnsafeAtomics.add!(ptr, Int32(1), Symbol(o)) == 2i - 1
        end
        @test UnsafeAtomics.add!(ptr, Int32(1), :acquire_release) == 10
        @test UnsafeAtomics.add!(ptr, Int32(1), :sequentially_consistent) == 11
        for i in 1:5
            @test @inferred(UnsafeAtomics.load(ptr, monotonic, pick_scope_symbol(i))) == 12
        end
        for i in 1:3
            @test @inferred(UnsafeAtomics.load(ptr, monotonic, pick_scope(i))) == 12
        end
        @test_throws Base.ConcurrencyViolationError UnsafeAtomics.add!(ptr, Int32(1), :unordered)
        @test_throws Base.ConcurrencyViolationError UnsafeAtomics.add!(ptr, Int32(1), :relaxed)
        @test_throws ArgumentError UnsafeAtomics.add!(ptr, Int32(1), monotonic, :agent)
        @test xs[1] == 12
    end
end

end  # module
