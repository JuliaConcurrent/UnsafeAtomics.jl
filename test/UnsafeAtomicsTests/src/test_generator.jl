module TestGenerator

using UnsafeAtomics: UnsafeAtomics
using UnsafeAtomics.Internal:
    llvm_load, llvm_store!, llvm_rmw!, llvm_cmpxchg!, llvm_fence, ATOMIC_SIZES, HAS_BFLOAT16
using Core: LLVMPtr
using InteractiveUtils: code_llvm
using Test
using Base: ConcurrencyViolationError

using ..Bits

# Two distinct values of every supported type.
values(::Type{T}) where {T<:Integer} = (T(1), T(2))
values(::Type{Bool}) = (false, true)
values(::Type{T}) where {T<:AbstractFloat} = (T(-0.0), T(2))
values(::Type{T}) where {T<:Union{Ptr,LLVMPtr}} = reinterpret.(T, (UInt(8), UInt(16)))
values(::Type{Bits32}) = (Bits32(0x00000001), Bits32(0x00000002))
if HAS_BFLOAT16
    # from memory: before LLVM 17, the AArch64 back-end can't materialize a bfloat constant
    const BFLOAT16_BITS = [0x8000, 0x4000]
    values(::Type{Core.BFloat16}) = Tuple(reinterpret(Core.BFloat16, BFLOAT16_BITS))
end

const TYPES = Any[
    Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, Bool,
    Float16, Float32, Float64, Ptr{Cvoid}, LLVMPtr{Cvoid,1}, Bits32,
]
16 in ATOMIC_SIZES && append!(TYPES, [Int128, UInt128])
HAS_BFLOAT16 && push!(TYPES, Core.BFloat16)

# The pointer kinds to run on (the CPU can only access address space 0).
pointers(xs) = (pointer(xs), reinterpret(LLVMPtr{eltype(xs),0}, pointer(xs)))

const NOMD = Val(())

function test_runtime()
    @testset for T in TYPES, i in 1:2
        a, b = values(T)
        xs = T[a, a]
        p = pointers(xs)[i]
        al = Val(sizeof(T))
        GC.@preserve xs begin
            @test llvm_load(p, Val(:acquire), Val(:system), Val(false), al, NOMD) === a
            llvm_store!(p, b, Val(:release), Val(:singlethread), Val(true), al, NOMD)
            @test xs[1] === b
            @test llvm_rmw!(p, Val(:xchg), a, Val(:acq_rel), Val(:workgroup), Val(false), al, NOMD) === b
            @test xs[1] === a
            @test llvm_cmpxchg!(p, a, b, Val(:seq_cst), Val(:seq_cst), Val(:device),
                                Val(false), Val(false), al, NOMD) === (old = a, success = true)
            @test llvm_cmpxchg!(p, a, a, Val(:monotonic), Val(:monotonic), Val(:system),
                                Val(false), Val(true), al, NOMD) === (old = b, success = false)
            @test xs == T[b, a]
        end
    end
end

function test_runtime_rmw()
    xs = Int32[5, 0]
    GC.@preserve xs for p in pointers(xs)
        rmw!(op, x) = llvm_rmw!(p, Val(op), Int32(x), Val(:monotonic), Val(:system), Val(false), Val(4), NOMD)
        xs[1] = 5
        @test rmw!(:add, 3) === Int32(5)
        @test rmw!(:sub, 1) === Int32(8)
        @test rmw!(:and, 6) === Int32(7)
        @test rmw!(:or, 8) === Int32(6)
        @test rmw!(:xor, 1) === Int32(14)
        @test rmw!(:nand, 3) === Int32(15)
        @test rmw!(:max, 1) === ~Int32(3)
        @test rmw!(:umax, 1) === Int32(1)
        @test rmw!(:min, -2) === Int32(1)
        @test rmw!(:umin, 0) === Int32(-2)
        @test xs[1] === Int32(0)
    end
    fs = Float32[1, 0]
    GC.@preserve fs for p in pointers(fs)
        rmw!(op, x) = llvm_rmw!(p, Val(op), Float32(x), Val(:monotonic), Val(:system), Val(false), Val(4), NOMD)
        fs[1] = 1
        @test rmw!(:fadd, 2) === 1f0
        @test rmw!(:fsub, 1) === 3f0
        @test rmw!(:fmax, 4) === 2f0
        @test rmw!(:fmin, 1) === 4f0
        @test fs[1] === 1f0
    end
end

function test_runtime_weak_cmpxchg()
    xs = Int64[1, 0]
    GC.@preserve xs for p in pointers(xs)
        xs[1] = 1
        # a weak cmpxchg may fail spuriously
        while !llvm_cmpxchg!(p, 1, 2, Val(:acq_rel), Val(:acquire), Val(:system),
                             Val(true), Val(false), Val(8), NOMD).success
        end
        @test xs[1] == 2
        @test llvm_cmpxchg!(p, 1, 3, Val(:acq_rel), Val(:acquire), Val(:system),
                            Val(true), Val(false), Val(8), NOMD) === (old = 2, success = false)
    end
end

function test_runtime_zero_size()
    xs = [nothing, nothing]
    GC.@preserve xs for p in pointers(xs)
        @test llvm_load(p, Val(:seq_cst), Val(:system), Val(false), Val(0), NOMD) === nothing
        @test llvm_store!(p, nothing, Val(:seq_cst), Val(:system), Val(false), Val(0), NOMD) === nothing
        @test llvm_rmw!(p, Val(:xchg), nothing, Val(:acq_rel), Val(:device), Val(false), Val(0), NOMD) === nothing
        @test llvm_cmpxchg!(p, nothing, nothing, Val(:acq_rel), Val(:acquire), Val(:system),
                            Val(false), Val(false), Val(0), NOMD) === (old = nothing, success = true)
    end
end

function test_runtime_fence()
    for scope in (:system, :singlethread, :workgroup, :agent), order in (:acquire, :release, :acq_rel, :seq_cst)
        @test llvm_fence(Val(order), Val(scope), NOMD) === nothing
    end
    @test llvm_fence(Val(:monotonic), Val(:system), NOMD) === nothing
end

# The generated code, without the counters that code coverage adds (`atomicrmw add` on a
# constant address).
llvm_ir(f, types; kwargs...) =
    join(filter(!contains("inttoptr ("),
                split(sprint(io -> code_llvm(io, f, types; debuginfo = :none, kwargs...)), '\n')), '\n')

# The line of the generated code that has the atomic instruction.
function instruction(f, types; raw = false)
    ir = llvm_ir(f, types; raw)
    lines = filter(split(ir, '\n')) do line
        occursin(r"\b(load atomic|store atomic|atomicrmw|cmpxchg|fence)\b", line) &&
            # GC safepoints in raw code
            !occursin("fence syncscope(\"singlethread\") seq_cst", line) &&
            !occursin("%safepoint", line)
    end
    return strip(only(lines))
end

# The textual form of pointers and values in the IR. LLVM prints opaque pointers from Julia
# 1.11 on, while llvmcall passes `Ptr` as an integer until 1.12.
const OPAQUE = VERSION >= v"1.11-"
ir_pointer(A, pointee) = OPAQUE ? (A == 0 ? "ptr" : "ptr addrspace($A)") :
                         (A == 0 ? "$pointee*" : "$pointee addrspace($A)*")
ir_type(::Type{Float16}) = "half"
ir_type(::Type{Float32}) = "float"
ir_type(::Type{Float64}) = "double"
ir_type(::Type{<:Ptr}) = VERSION >= v"1.12-" ? "ptr" : "i64"
ir_type(::Type{<:LLVMPtr{<:Any,A}}) where {A} = ir_pointer(A, "i8")
ir_type(::Type{T}) where {T} = HAS_BFLOAT16 && T === Core.BFloat16 ? "bfloat" : "i$(8sizeof(T))"
ir_cmpxchg_type(::Type{T}) where {T} =
    T <: AbstractFloat ? "i$(8sizeof(T))" : ir_type(T)

addrspace(::Type{<:Ptr}) = 0
addrspace(::Type{<:LLVMPtr{<:Any,A}}) where {A} = A

gen_load(p, ::Val{o}, ::Val{s}, ::Val{v}, ::Val{al}) where {o,s,v,al} =
    llvm_load(p, Val(o), Val(s), Val(v), Val(al), NOMD)
gen_store!(p, x, ::Val{o}, ::Val{s}, ::Val{v}, ::Val{al}) where {o,s,v,al} =
    llvm_store!(p, x, Val(o), Val(s), Val(v), Val(al), NOMD)
gen_rmw!(p, x, ::Val{op}, ::Val{o}, ::Val{s}, ::Val{v}, ::Val{al}) where {op,o,s,v,al} =
    llvm_rmw!(p, Val(op), x, Val(o), Val(s), Val(v), Val(al), NOMD)
gen_cmpxchg!(p, c, n, ::Val{so}, ::Val{fo}, ::Val{s}, ::Val{w}, ::Val{v}, ::Val{al}) where {so,fo,s,w,v,al} =
    llvm_cmpxchg!(p, c, n, Val(so), Val(fo), Val(s), Val(w), Val(v), Val(al), NOMD)

scope_ir(s) = s === :system ? "" : " syncscope(\"$s\")"
volatile_ir(v) = v ? " volatile" : ""

function test_ir()
    @testset for T in TYPES, P in (Ptr{T}, LLVMPtr{T,0}, LLVMPtr{T,1}, LLVMPtr{T,3})
        lt, ct = ir_type(T), ir_cmpxchg_type(T)
        A = addrspace(P)
        al = sizeof(T)
        pt(t) = ir_pointer(A, t) * " %"
        for (s, v) in ((:system, false), (:workgroup, true))
            # Julia's intrinsics load and store floats as integers
            intrinsic = UnsafeAtomics.Internal.use_intrinsics(P, T, s, v, al, ())
            lt = intrinsic ? ct : ir_type(T)
            ld = instruction(gen_load, Tuple{P,Val{:acquire},Val{s},Val{v},Val{al}})
            @test startswith(ld, r"%\S+ = load atomic" * volatile_ir(v) * " $lt, " * pt(lt))
            @test endswith(ld, scope_ir(s) * " acquire, align $al")

            st = instruction(gen_store!, Tuple{P,T,Val{:release},Val{s},Val{v},Val{al}})
            @test startswith(st, "store atomic" * volatile_ir(v) * " $lt %")
            @test occursin(", " * pt(lt), st)
            @test endswith(st, scope_ir(s) * " release, align $al")

            rmw = instruction(gen_rmw!, Tuple{P,T,Val{:xchg},Val{:acq_rel},Val{s},Val{v},Val{al}})
            lt = ir_type(T)
            @test startswith(rmw, r"%\S+ = atomicrmw" * volatile_ir(v) * " xchg " * pt(lt))
            @test occursin(", $lt %", rmw)
            @test endswith(rmw, scope_ir(s) * " acq_rel, align $al")

            for w in (false, true)
                cas = instruction(gen_cmpxchg!, Tuple{P,T,T,Val{:acq_rel},Val{:acquire},Val{s},Val{w},Val{v},Val{al}})
                @test startswith(cas, r"%\S+ = cmpxchg" * (w ? " weak" : "") * volatile_ir(v) * " " * pt(ct))
                @test occursin(", $ct %", cas)
                @test endswith(cas, scope_ir(s) * " acq_rel acquire, align $al")
            end
        end
        # no casts through integers or memory
        ir = llvm_ir(gen_cmpxchg!, Tuple{P,T,T,Val{:acq_rel},Val{:acquire},Val{:device},Val{false},Val{false},Val{al}})
        @test !occursin("alloca", ir)
        P <: LLVMPtr && @test !occursin("inttoptr", ir)
    end
end

function test_ir_orderings()
    P = LLVMPtr{Int32,1}
    for o in (:unordered, :monotonic, :acquire, :seq_cst)
        @test endswith(instruction(gen_load, Tuple{P,Val{o},Val{:device},Val{false},Val{4}}), " $o, align 4")
    end
    for o in (:unordered, :monotonic, :release, :seq_cst)
        @test endswith(instruction(gen_store!, Tuple{P,Int32,Val{o},Val{:device},Val{false},Val{4}}), " $o, align 4")
    end
    for o in (:monotonic, :acquire, :release, :acq_rel, :seq_cst)
        @test endswith(instruction(gen_rmw!, Tuple{P,Int32,Val{:add},Val{o},Val{:device},Val{false},Val{4}}), " $o, align 4")
        for fo in (:monotonic, :acquire, :seq_cst)
            cas = instruction(gen_cmpxchg!, Tuple{P,Int32,Int32,Val{o},Val{fo},Val{:device},Val{false},Val{false},Val{4}})
            @test endswith(cas, " $o $fo, align 4")
        end
    end
    # Julia's names
    @test endswith(instruction(gen_rmw!, Tuple{P,Int32,Val{:add},Val{:acquire_release},Val{:device},Val{false},Val{4}}), " acq_rel, align 4")
    @test endswith(instruction(gen_rmw!, Tuple{P,Int32,Val{:add},Val{:sequentially_consistent},Val{:device},Val{false},Val{4}}), " seq_cst, align 4")
end

# which calls use `Core.Intrinsics` instead of an llvmcall
function intrinsic(f, types)
    src = string(first(only(Base.code_typed(f, types))))
    has_intrinsic = occursin("Core.Intrinsics.atomic_pointer", src) ||
                    occursin("atomic_pointerref", src) || occursin("atomic_pointerset", src) ||
                    occursin("atomic_pointerreplace", src)
    has_llvmcall = occursin("llvmcall", src)
    @assert has_intrinsic != has_llvmcall
    return has_intrinsic
end

function test_intrinsics()
    # Ptr in the system scope, with the defaults: the same as UnsafeAtomics 0.3
    for T in (Int8, Int32, UInt64, Float16, Float32, Float64)
        P = Ptr{T}
        @test intrinsic(gen_load, Tuple{P,Val{:acquire},Val{:system},Val{false},Val{sizeof(T)}})
        @test intrinsic(gen_store!, Tuple{P,T,Val{:release},Val{:system},Val{false},Val{sizeof(T)}})
        @test intrinsic(gen_cmpxchg!, Tuple{P,T,T,Val{:acq_rel},Val{:acquire},Val{:system},Val{false},Val{false},Val{sizeof(T)}})
        @test !intrinsic(gen_rmw!, Tuple{P,T,Val{:xchg},Val{:acq_rel},Val{:system},Val{false},Val{sizeof(T)}})
    end
    P = Ptr{Int32}
    # what the intrinsics can't express
    @test !intrinsic(gen_load, Tuple{P,Val{:acquire},Val{:singlethread},Val{false},Val{4}})
    @test !intrinsic(gen_load, Tuple{P,Val{:acquire},Val{:system},Val{true},Val{4}})
    @test !intrinsic(gen_load, Tuple{P,Val{:acquire},Val{:system},Val{false},Val{8}})
    @test !intrinsic(gen_store!, Tuple{P,Int32,Val{:release},Val{:system},Val{true},Val{4}})
    @test !intrinsic(gen_cmpxchg!, Tuple{P,Int32,Int32,Val{:acq_rel},Val{:acquire},Val{:system},Val{true},Val{false},Val{4}})
    @test !intrinsic(gen_cmpxchg!, Tuple{P,Int32,Int32,Val{:monotonic},Val{:seq_cst},Val{:system},Val{false},Val{false},Val{4}})
    @test intrinsic(gen_cmpxchg!, Tuple{P,Int32,Int32,Val{:release},Val{:acquire},Val{:system},Val{false},Val{false},Val{4}})
    # other types and pointers
    @test !intrinsic(gen_load, Tuple{Ptr{Bool},Val{:acquire},Val{:system},Val{false},Val{1}})
    @test !intrinsic(gen_load, Tuple{Ptr{Ptr{Cvoid}},Val{:acquire},Val{:system},Val{false},Val{8}})
    @test !intrinsic(gen_load, Tuple{LLVMPtr{Int32,0},Val{:acquire},Val{:system},Val{false},Val{4}})

    # both compare bitwise
    xs = Float32[NaN, -0.0]
    GC.@preserve xs for scope in (:system, :singlethread)
        cas(i, c, n) = llvm_cmpxchg!(pointer(xs, i), c, n, Val(:seq_cst), Val(:seq_cst), Val(scope), Val(false), Val(false), Val(4), NOMD)
        @test cas(1, NaN32, NaN32) === (old = NaN32, success = true)
        @test cas(2, 0f0, 1f0) === (old = -0f0, success = false)
    end
end

gen_fence(::Val{o}, ::Val{s}) where {o,s} = llvm_fence(Val(o), Val(s), NOMD)

function test_ir_scopes_and_alignment()
    P = LLVMPtr{Int32,1}
    for s in (:singlethread, :subgroup, :workgroup, :device, :agent, :system)
        @test endswith(instruction(gen_load, Tuple{P,Val{:acquire},Val{s},Val{false},Val{4}}),
                       scope_ir(s) * " acquire, align 4")
        s === :singlethread && continue  # a GC safepoint looks the same
        @test instruction(gen_fence, Tuple{Val{:seq_cst},Val{s}}) == "fence" * scope_ir(s) * " seq_cst"
    end
    @test endswith(instruction(gen_load, Tuple{P,Val{:acquire},Val{:device},Val{false},Val{16}}), "align 16")
    @test endswith(instruction(gen_rmw!, Tuple{P,Int32,Val{:add},Val{:monotonic},Val{:device},Val{false},Val{8}}), "align 8")
end

gen_md_rmw!(p, x, ::Val{md}) where {md} =
    llvm_rmw!(p, Val(:add), x, Val(:monotonic), Val(:workgroup), Val(false), Val(4), Val(md))
gen_md_fence(::Val{md}) where {md} = llvm_fence(Val(:release), Val(:workgroup), Val(md))

function test_ir_metadata()
    tag = (Symbol("metal-synchronize-as"), :threadgroup)
    tag2 = (Symbol("metal-synchronize-as"), :device)
    P = LLVMPtr{Int32,1}
    # `code_llvm` hides metadata unless raw
    for (md, nodes) in (((:mmra, (tag,)),) => [r"^!\d+ = !{!\"metal-synchronize-as\", !\"threadgroup\"}$"],
                        ((:mmra, (tag, tag2)),) => [r"^!\d+ = !{!\d+, !\d+}$",
                                                    r"^!\d+ = !{!\"metal-synchronize-as\", !\"device\"}$"])
        for (f, types) in ((gen_md_rmw!, Tuple{P,Int32,Val{md}}), (gen_md_fence, Tuple{Val{md}}))
            ir = sprint(io -> code_llvm(io, f, types; debuginfo = :none, raw = true, dump_module = true))
            @test occursin(r", !mmra !\d+", instruction(f, types; raw = true))
            for node in nodes
                @test any(line -> occursin(node, line), split(ir, '\n'))
            end
        end
    end
    @test !occursin("mmra", instruction(gen_md_rmw!, Tuple{P,Int32,Val{()}}; raw = true))
    @test !occursin("mmra", instruction(gen_md_rmw!, Tuple{P,Int32,Val{((:mmra, ()),)}}; raw = true))
end

function test_invalid()
    xs = Int32[1]
    p = pointer(xs)
    V = Val
    GC.@preserve xs begin
        for o in (:release, :acq_rel, :bogus)
            @test_throws ConcurrencyViolationError llvm_load(p, V(o), V(:system), V(false), V(4), NOMD)
        end
        for o in (:acquire, :acq_rel)
            @test_throws ConcurrencyViolationError llvm_store!(p, Int32(1), V(o), V(:system), V(false), V(4), NOMD)
        end
        @test_throws ConcurrencyViolationError llvm_rmw!(p, V(:add), Int32(1), V(:unordered), V(:system), V(false), V(4), NOMD)
        for (so, fo) in ((:unordered, :monotonic), (:monotonic, :release), (:seq_cst, :acq_rel), (:seq_cst, :unordered))
            @test_throws ConcurrencyViolationError llvm_cmpxchg!(p, Int32(1), Int32(2), V(so), V(fo), V(:system), V(false), V(false), V(4), NOMD)
        end
        @test_throws ConcurrencyViolationError llvm_fence(V(:unordered), V(:system), NOMD)

        for al in (0, 2, 3, 6, 4.0)
            @test_throws ArgumentError llvm_load(p, V(:monotonic), V(:system), V(false), V(al), NOMD)
        end
        @test_throws ArgumentError llvm_load(p, V(:monotonic), V(1), V(false), V(4), NOMD)
        @test_throws ArgumentError llvm_load(p, V(:monotonic), V(:system), V(1), V(4), NOMD)
        @test_throws ArgumentError llvm_cmpxchg!(p, Int32(1), Int32(2), V(:monotonic), V(:monotonic), V(:system), V(nothing), V(false), V(4), NOMD)
        @test_throws ArgumentError llvm_rmw!(p, V(:mul), Int32(1), V(:monotonic), V(:system), V(false), V(4), NOMD)
        # operations that don't apply to the type
        @test_throws ArgumentError llvm_rmw!(p, V(:fadd), Int32(1), V(:monotonic), V(:system), V(false), V(4), NOMD)
        fs = Float32[1]
        GC.@preserve fs for op in (:add, :and, :umax)
            @test_throws ArgumentError llvm_rmw!(pointer(fs), V(op), 1f0, V(:monotonic), V(:system), V(false), V(4), NOMD)
        end
        ps = [C_NULL]
        GC.@preserve ps begin
            @test_throws ArgumentError llvm_rmw!(pointer(ps), V(:add), C_NULL, V(:monotonic), V(:system), V(false), V(8), NOMD)
            @test llvm_rmw!(pointer(ps), V(:xchg), C_NULL, V(:monotonic), V(:system), V(false), V(8), NOMD) === C_NULL
        end
        @test_throws ArgumentError llvm_load(p, V(:monotonic), V(:system), V(false), V(4), V(((:tbaa, ()),)))
        @test_throws ArgumentError llvm_load(p, V(:monotonic), V(:system), V(false), V(4), V(((:mmra, ((:a, 1),)),)))
        @test xs == Int32[1]
    end
    # unsupported value types
    ys = [(1, 2)]
    GC.@preserve ys begin
        @test_throws ArgumentError llvm_load(pointer(ys), V(:monotonic), V(:system), V(false), V(16), NOMD)
    end
end

end  # module
