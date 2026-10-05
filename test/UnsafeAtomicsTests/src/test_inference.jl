module TestInference

using UnsafeAtomics: UnsafeAtomics, unordered, monotonic, acquire, release, acq_rel, seq_cst
using UnsafeAtomics: singlethread, subgroup, workgroup, device, system
using Core: LLVMPtr
using Test

using ..Helpers

# Constant orderings and scopes, as objects or as Symbols, give a single instruction.
add_constant(p, x) = UnsafeAtomics.add!(p, x, acquire, device)
add_symbols(p, x) = UnsafeAtomics.add!(p, x, :acquire, :device)
add_alias(p, x) = UnsafeAtomics.add!(p, x, :acquire_release, :workgroup)
add_agent(p, x) = UnsafeAtomics.add!(p, x, monotonic, UnsafeAtomics.SyncScope(:agent))
modify_symbols(p, x) = UnsafeAtomics.modify!(p, +, x, :seq_cst, :subgroup)
load_symbols(p) = UnsafeAtomics.load(p, :sequentially_consistent, :singlethread)
store_symbols(p, x) = UnsafeAtomics.store!(p, x, :release, :system)
cas_symbols(p, c, n) = UnsafeAtomics.cas!(p, c, n, :acq_rel, :acquire, :device)
cas_acq_rel(p, c, n) = UnsafeAtomics.cas!(p, c, n, :acq_rel)
cas_acquire_release(p, c, n) = UnsafeAtomics.cas!(p, c, n, :acquire_release)
cas_release(p, c, n) = UnsafeAtomics.cas!(p, c, n, :release)
cas_seq_cst(p, c, n) = UnsafeAtomics.cas!(p, c, n, :sequentially_consistent)
fence_symbols() = UnsafeAtomics.fence(:acquire_release, :workgroup)

const P = LLVMPtr{Int32,1}

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
    for (f, orders) in ((cas_acq_rel, "acq_rel acquire"), (cas_acquire_release, "acq_rel acquire"),
                        (cas_release, "release monotonic"), (cas_seq_cst, "seq_cst seq_cst"))
        check(f, Tuple{P,Int32,Int32}, "cmpxchg", " $orders, align 4",
              @NamedTuple{old::Int32, success::Bool})
    end
    check(fence_symbols, Tuple{}, r"^\s*fence ", "fence syncscope(\"workgroup\") acq_rel", Nothing)
end

kw_load(p) = UnsafeAtomics.load(p, acquire, device; volatile = true, align = 16)
kw_store(p, x) = UnsafeAtomics.store!(p, x, release, device; volatile = true)
kw_cas(p, c, n) = UnsafeAtomics.cas!(p, c, n, acq_rel, acquire, device; weak = true, align = 8)
kw_modify(p, x) = UnsafeAtomics.modify!(p, +, x, monotonic, device; volatile = true, align = 8)
kw_add(p, x) = UnsafeAtomics.add!(p, x; volatile = true)
kw_max_cas(p, x) = UnsafeAtomics.modify!(p, *, x, monotonic, device; volatile = true, align = 8)
# Constant keywords only reach a wrapper that Julia inlines.
@inline kw_inlined(p; kwargs...) = UnsafeAtomics.load(p, acquire, device; kwargs...)
kw_forward_inlined(p) = kw_inlined(p; volatile = true, align = 8)

function test_keywords()
    function instruction(f, types)
        ir = llvm_ir(f, types)
        @test !occursin(r"apply_generic|jl_invoke|jl_f_", ir)
        return [strip(l) for l in split(ir, '\n') if occursin(r"atomic|cmpxchg", l) && !occursin("tag_addr", l)]
    end
    @test endswith(only(instruction(kw_load, Tuple{P})), "syncscope(\"device\") acquire, align 16")
    @test occursin("load atomic volatile", only(instruction(kw_load, Tuple{P})))
    @test occursin("store atomic volatile", only(instruction(kw_store, Tuple{P,Int32})))
    cas = only(instruction(kw_cas, Tuple{P,Int32,Int32}))
    @test occursin("cmpxchg weak", cas) && endswith(cas, "acq_rel acquire, align 8")
    @test occursin(r"atomicrmw volatile add .* align 8$", only(instruction(kw_modify, Tuple{P,Int32})))
    @test occursin(r"atomicrmw volatile add .* seq_cst, align 4$", only(instruction(kw_add, Tuple{P,Int32})))
    # the compare-and-swap loop passes them on
    loop = instruction(kw_max_cas, Tuple{P,Int32})
    @test length(loop) >= 2
    @test all(l -> occursin("volatile", l) && endswith(l, "align 8"), loop)
    @test endswith(only(instruction(kw_forward_inlined, Tuple{P})), "acquire, align 8")

    xs = Int64[1, 2]
    GC.@preserve xs begin
        ptr = pointer(xs)
        @test UnsafeAtomics.load(ptr; volatile = true, align = 16) == 1
        @test UnsafeAtomics.cas!(ptr, 1, 3; weak = false) === (old = 1, success = true)
        @test_throws ArgumentError UnsafeAtomics.load(ptr; align = 4)
        @test_throws ArgumentError UnsafeAtomics.store!(ptr, 1; align = 12)
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
            @test UnsafeAtomics.load(ptr; volatile = isodd(i)) == 10
        end
        @test UnsafeAtomics.cas!(ptr, Int32(10), Int32(12), pick_order(4), pick_order(2)) ===
              (old = Int32(10), success = true)
        @test_throws Base.ConcurrencyViolationError UnsafeAtomics.add!(ptr, Int32(1), Symbol("unordered"))
        @test xs[1] == 12
    end
end

end  # module
