# Atomic instructions as `llvmcall`s.
#
# Each primitive below compiles to exactly one LLVM atomic instruction, which LLVM.jl builds
# for the pointer type, value type, ordering, syncscope, flags and metadata. All of those are
# type parameters, so these primitives don't depend on constant propagation; they are the
# layer that back-ends can rely on:
#
#     llvm_load(ptr, Val(order), Val(scope), Val(volatile), Val(align), Val(md)) -> T
#     llvm_store!(ptr, x, Val(order), Val(scope), Val(volatile), Val(align), Val(md))
#     llvm_rmw!(ptr, Val(op), x, Val(order), Val(scope), Val(volatile), Val(align), Val(md)) -> T
#     llvm_cmpxchg!(ptr, cmp, new, Val(success_order), Val(failure_order), Val(scope),
#                   Val(weak), Val(volatile), Val(align), Val(md)) -> (; old::T, success::Bool)
#     llvm_fence(Val(order), Val(scope), Val(md))
#
# Orderings are LLVM's (or Julia's) names, e.g. `:acq_rel` or `:acquire_release`. The scope
# is an LLVM syncscope name, with `:system` for the default scope, `op` an `atomicrmw`
# operation, and `md` a tuple of metadata to attach, `((:mmra, ((prefix, suffix), ...)),)`.
# Invalid orderings throw a `ConcurrencyViolationError`, other invalid arguments an
# `ArgumentError`, when the primitive is called.

const AnyPtr{T} = Union{Ptr{T},LLVMPtr{T}}

address_space(::Type{<:Ptr}) = 0
address_space(::Type{<:LLVMPtr{<:Any,A}}) where {A} = A

const HAS_BFLOAT16 = isdefined(Core, :BFloat16)

is_ieee_float(T) =
    T === Float16 || T === Float32 || T === Float64 || (HAS_BFLOAT16 && T === Core.BFloat16)

# Like Base, only use 128-bit atomics where they are known to work
# (https://github.com/JuliaLang/julia/blob/v1.6.3/base/atomics.jl#L23-L30).
const ATOMIC_SIZES =
    if Sys.ARCH == :i686 || startswith(string(Sys.ARCH), "arm") ||
       Sys.ARCH === :powerpc64le || Sys.ARCH === :ppc64le
        (1, 2, 4, 8)
    else
        (1, 2, 4, 8, 16)
    end

# Whether values of type `T` can be accessed atomically. Their LLVM type is what `llvmcall`
# lowers `T` to: `iN` for integers and other primitive types, `half`, `float`, `double` and
# `bfloat` for floats, and a pointer or (for `Ptr` before Julia 1.12) an integer for pointers.
is_atomic_type(::Type{T}) where {T} = isprimitivetype(T) && sizeof(T) in ATOMIC_SIZES

is_zero_size(T) = Base.issingletontype(T)

# The pointer argument as a pointer to values of LLVM type `ty`. Before Julia 1.12, llvmcall
# passes a `Ptr` as an integer, and a `Core.LLVMPtr` as an `i8` pointer; with opaque
# pointers, the cast is folded away.
function pointer_operand!(builder, ptr, ty, A)
    pt = LLVM.PointerType(ty, A)
    return ptr.value_type isa LLVM.IntegerType ? inttoptr!(builder, ptr, pt) :
           bitcast!(builder, ptr, pt)
end

julia_order(o) = o === :acq_rel ? :acquire_release : o === :seq_cst ? :sequentially_consistent : o
order_strength(o) = findfirst(==(o), (:unordered, :monotonic, :acquire, :release, :acq_rel, :seq_cst))

# The LLVM.jl ordering for the name of a valid ordering.
atomic_ordering(o::Symbol) = parse(LLVM.AtomicOrdering.T, String(o))

# Julia's pointer intrinsics only take 8 bytes before 1.12, and on 32-bit platforms.
const MAX_POINTERATOMIC_SIZE = VERSION >= v"1.12.0-DEV.161" && Int == Int64 ? 16 : 8

# Atomics on Ptr in the system scope keep using Julia's intrinsics where those can express
# them: they compile to the same instruction, but Julia's compiler knows what they do.
use_intrinsics(P, T, scope, volatile, align, md) =
    P <: Ptr && scope === :system && !volatile && align == sizeof(T) && md === () &&
    (T <: Base.BitInteger || T === Float16 || T === Float32 || T === Float64) &&
    sizeof(T) <= MAX_POINTERATOMIC_SIZE

const LOAD_ORDERS = (:unordered, :monotonic, :acquire, :seq_cst)
const STORE_ORDERS = (:unordered, :monotonic, :release, :seq_cst)
const RMW_ORDERS = (:monotonic, :acquire, :release, :acq_rel, :seq_cst)
const CAS_FAILURE_ORDERS = (:monotonic, :acquire, :seq_cst)
const FENCE_ORDERS = (:acquire, :release, :acq_rel, :seq_cst)

# The `atomicrmw` operation called `op` in LLVM IR, or `nothing` if this version of LLVM
# doesn't have it. LLVM.jl knows every operation, on every version of LLVM.
function rmw_operation(op)
    op isa Symbol || return nothing
    binop = tryparse(LLVM.AtomicRMWBinOp.T, String(op))
    return binop !== nothing && LLVM.isavailable(binop) ? binop : nothing
end

# Invalid arguments are reported when the primitive is called, not when it's generated.
struct InvalidOrdering <: Exception end
struct InvalidArgument <: Exception
    msg::String
end
invalid(msg...) = throw(InvalidArgument(string(msg...)))

function generate(generator, args...)
    try
        return generator(args...)
    catch err
        err isa InvalidOrdering && return :(throw_invalid_ordering())
        err isa InvalidArgument && return :(throw(ArgumentError($(err.msg))))
        rethrow()
    end
end

function check_order(o, valid)
    o = normalize_order(o)
    o in valid || throw(InvalidOrdering())
    return o
end

function check_access(::Type{T}, align, volatile, weak = false) where {T}
    volatile isa Bool || invalid("volatile must be a Bool, got ", repr(volatile))
    weak isa Bool || invalid("weak must be a Bool, got ", repr(weak))
    is_zero_size(T) && return
    is_atomic_type(T) || invalid("unsupported atomic type ", T)
    align isa Integer && !(align isa Bool) && ispow2(align) && align >= sizeof(T) ||
        invalid("invalid alignment ", repr(align), " for an atomic ", T,
                ": expected a power of two, at least ", sizeof(T))
    return
end

# The scope for LLVM.jl's builders, which also call the default scope "system".
function check_scope(scope)
    scope isa Symbol || invalid("invalid syncscope ", repr(scope))
    return String(scope)
end

function check_metadata(md)
    md isa Tuple || invalid("invalid metadata ", repr(md))
    for entry in md
        entry isa Tuple{Symbol,Tuple} && entry[1] === :mmra ||
            invalid("unsupported metadata ", repr(entry))
        tags = entry[2]
        all(tag -> tag isa Tuple{Symbol,Symbol}, tags) || invalid("invalid MMRA tags ", repr(tags))
    end
    return md
end

function attach_metadata!(inst, md)
    for (_, tags) in md
        isempty(tags) && continue
        mmra!(inst, (String(prefix) => String(suffix) for (prefix, suffix) in tags)...)
    end
    return inst
end

# An atomic operation on a zero-size value only has to order memory.
fence_for(o, scope, md) =
    o === :unordered || o === :monotonic ? :(nothing) :
    :(llvm_fence(Val($(QuoteNode(o))), Val($(QuoteNode(scope))), Val($md)))

function fence_ir(order, scope, md)
    order = normalize_order(order)
    # like `Core.Intrinsics.atomic_fence`; LLVM would reject it
    order === :monotonic && return :(nothing)
    o = check_order(order, FENCE_ORDERS)
    s, md = check_scope(scope), check_metadata(md)
    return generate_llvmcall(Nothing, Tuple{}) do builder
        attach_metadata!(fence!(builder, atomic_ordering(o); scope = s), md)
        nothing
    end
end

function load_ir(P, T, order, scope, volatile, align, md)
    o = check_order(order, LOAD_ORDERS)
    check_access(T, align, volatile)
    s, md = check_scope(scope), check_metadata(md)
    is_zero_size(T) && return :($(fence_for(o, scope, md)); $(T.instance))
    use_intrinsics(P, T, scope, volatile, align, md) &&
        return :(Core.Intrinsics.atomic_pointerref(ptr, $(QuoteNode(julia_order(o)))))
    return generate_llvmcall(T, Tuple{P}, :ptr) do builder, ptr
        lt = convert(LLVMType, T)
        p = pointer_operand!(builder, ptr, lt, address_space(P))
        attach_metadata!(load!(builder, lt, p; ordering = atomic_ordering(o), scope = s,
                               align, volatile), md)
    end
end

function store_ir(P, T, order, scope, volatile, align, md)
    o = check_order(order, STORE_ORDERS)
    check_access(T, align, volatile)
    s, md = check_scope(scope), check_metadata(md)
    is_zero_size(T) && return :($(fence_for(o, scope, md)); nothing)
    use_intrinsics(P, T, scope, volatile, align, md) &&
        return :(Core.Intrinsics.atomic_pointerset(ptr, x, $(QuoteNode(julia_order(o)))); nothing)
    return generate_llvmcall(Nothing, Tuple{P,T}, :ptr, :x) do builder, ptr, x
        p = pointer_operand!(builder, ptr, x.value_type, address_space(P))
        attach_metadata!(store!(builder, x, p; ordering = atomic_ordering(o), scope = s,
                                align, volatile), md)
        nothing
    end
end

function rmw_ir(P, T, op, order, scope, volatile, align, md)
    o = check_order(order, RMW_ORDERS)
    check_access(T, align, volatile)
    binop = rmw_operation(op)
    binop === nothing &&
        invalid("unsupported atomicrmw operation ", repr(op), " on LLVM ", Base.libllvm_version)
    s, md = check_scope(scope), check_metadata(md)
    is_zero_size(T) && return :($(fence_for(o, scope, md)); $(T.instance))
    # floating-point operations only apply to floating-point values, and the others (except
    # `xchg`) to integers. values of a `Ptr` are integers before Julia 1.12, but only
    # support `xchg`.
    op !== :xchg &&
        (T <: Ptr || T <: LLVMPtr || LLVM.isfloatingpoint(binop) != is_ieee_float(T)) &&
        invalid("atomicrmw ", op, " doesn't apply to values of type ", T)
    return generate_llvmcall(T, Tuple{P,T}, :ptr, :x) do builder, ptr, x
        p = pointer_operand!(builder, ptr, x.value_type, address_space(P))
        inst = atomic_rmw!(builder, binop, p, x, atomic_ordering(o); scope = s, align, volatile)
        attach_metadata!(inst, md)
    end
end

function cmpxchg_ir(P, T, success_order, failure_order, scope, weak, volatile, align, md)
    so = check_order(success_order, RMW_ORDERS)
    fo = check_order(failure_order, CAS_FAILURE_ORDERS)
    check_access(T, align, volatile, weak)
    s, md = check_scope(scope), check_metadata(md)
    is_zero_size(T) &&
        return :($(fence_for(so, scope, md)); (old = $(T.instance), success = true))
    # the intrinsic is strong, and rejects a failure ordering stronger than the success one
    if !weak && use_intrinsics(P, T, scope, volatile, align, md) &&
       order_strength(fo) <= order_strength(so)
        return :(Core.Intrinsics.atomic_pointerreplace(
            ptr, cmp, new, $(QuoteNode(julia_order(so))), $(QuoteNode(julia_order(fo)))))
    end
    RT = @NamedTuple{old::T, success::Bool}
    return generate_llvmcall(RT, Tuple{P,T,T}, :ptr, :cmp, :new) do builder, ptr, cmp, new
        lt = cmp.value_type
        # cmpxchg only takes integers and pointers
        ct = is_ieee_float(T) ? LLVM.IntType(8 * sizeof(T)) : lt
        p = pointer_operand!(builder, ptr, ct, address_space(P))
        c, n = bitcast!(builder, cmp, ct), bitcast!(builder, new, ct)
        r = atomic_cmpxchg!(builder, p, c, n, atomic_ordering(so), atomic_ordering(fo);
                            scope = s, align, volatile, weak)
        attach_metadata!(r, md)
        old = bitcast!(builder, extract_value!(builder, r, 0), lt)
        success = zext!(builder, extract_value!(builder, r, 1), LLVM.Int8Type())
        # llvmcall lowers a pair of `i8`s to an array, and others to a struct
        rt = convert(LLVMType, RT)
        insert_value!(builder, insert_value!(builder, UndefValue(rt), old, 0), success, 1)
    end
end

@generated llvm_fence(::Val{order}, ::Val{scope}, ::Val{md}) where {order,scope,md} =
    generate(fence_ir, order, scope, md)

@generated llvm_load(
    ptr::AnyPtr{T}, ::Val{order}, ::Val{scope}, ::Val{volatile}, ::Val{align}, ::Val{md},
) where {T,order,scope,volatile,align,md} =
    generate(load_ir, ptr, T, order, scope, volatile, align, md)

@generated llvm_store!(
    ptr::AnyPtr{T}, x::T, ::Val{order}, ::Val{scope}, ::Val{volatile}, ::Val{align}, ::Val{md},
) where {T,order,scope,volatile,align,md} =
    generate(store_ir, ptr, T, order, scope, volatile, align, md)

@generated llvm_rmw!(
    ptr::AnyPtr{T}, ::Val{op}, x::T, ::Val{order}, ::Val{scope}, ::Val{volatile}, ::Val{align},
    ::Val{md},
) where {T,op,order,scope,volatile,align,md} =
    generate(rmw_ir, ptr, T, op, order, scope, volatile, align, md)

@generated llvm_cmpxchg!(
    ptr::AnyPtr{T}, cmp::T, new::T, ::Val{success_order}, ::Val{failure_order}, ::Val{scope},
    ::Val{weak}, ::Val{volatile}, ::Val{align}, ::Val{md},
) where {T,success_order,failure_order,scope,weak,volatile,align,md} =
    generate(cmpxchg_ir, ptr, T, success_order, failure_order, scope, weak, volatile, align, md)

# Read-modify-write with any function: an `atomicrmw` where one implements `op` on values of
# type `T`, and a compare-and-swap loop otherwise.
function native_rmw(@nospecialize(op), @nospecialize(T))
    is_zero_size(T) && return op === right ? :xchg : nothing
    is_atomic_type(T) || return nothing
    int = T <: Base.BitInteger
    # before LLVM 20, the AArch64 back-end can't compile floating-point atomicrmw on bfloat
    float = is_ieee_float(T) &&
            !(HAS_BFLOAT16 && T === Core.BFloat16 && Base.libllvm_version < v"20")
    bool = T === Bool
    if op === right
        return :xchg
    elseif op === (+)
        return int ? :add : float ? :fadd : nothing
    elseif op === (-)
        return int ? :sub : float ? :fsub : nothing
    elseif op === (&)
        return int || bool ? :and : nothing
    elseif op === (|)
        return int || bool ? :or : nothing
    elseif op === xor
        return int || bool ? :xor : nothing
    elseif op === (⊼)
        # a bitwise nand of Bools isn't a Bool
        return int ? :nand : nothing
    elseif op === max
        # Julia's `max` propagates NaNs and orders -0.0 before 0.0, like LLVM's `fmaximum`
        return T <: Base.BitSigned ? :max : T <: Base.BitUnsigned || bool ? :umax :
               float && LLVM.isavailable(LLVM.AtomicRMWBinOp.FMaximum) ? :fmaximum : nothing
    elseif op === min
        return T <: Base.BitSigned ? :min : T <: Base.BitUnsigned || bool ? :umin :
               float && LLVM.isavailable(LLVM.AtomicRMWBinOp.FMinimum) ? :fminimum : nothing
    elseif op === UnsafeAtomics.fmax
        return float ? :fmax : nothing
    elseif op === UnsafeAtomics.fmin
        return float ? :fmin : nothing
    end
    # LLVM parses these from version 16 or 20, but before LLVM 22 the AArch64 back-end can't
    # compile them. Back-ends for targets that can use `llvm_rmw!` directly.
    rmw = op === UnsafeAtomics.inc_wrap ? :uinc_wrap :
          op === UnsafeAtomics.dec_wrap ? :udec_wrap :
          op === UnsafeAtomics.sub_cond ? :usub_cond :
          op === UnsafeAtomics.sub_sat ? :usub_sat : nothing
    rmw !== nothing && T <: Base.BitUnsigned && Base.libllvm_version >= v"22" && return rmw
    return nothing
end

@inline function cas_loop!(ptr::AnyPtr{T}, op, x, order, scope, volatile, align, md) where {T}
    old = llvm_load(ptr, Val(:monotonic), scope, volatile, align, md)
    while true
        new = op(old, x)::T
        (; old, success) = llvm_cmpxchg!(
            ptr, old, new, order, Val(:monotonic), scope, Val(true), volatile, align, md)
        success && return old => new
    end
end

function modify_ir(T, op, order, fetch)
    check_order(order, RMW_ORDERS)
    rmw = Base.issingletontype(op) ? native_rmw(op.instance, T) : nothing
    if rmw === nothing
        loop = :(cas_loop!(ptr, op, x, order, scope, volatile, align, md))
        return fetch ? :(first($loop)) : loop
    end
    rmw = :(llvm_rmw!(ptr, Val($(QuoteNode(rmw))), x, order, scope, volatile, align, md))
    return fetch ? rmw : :(old = $rmw; old => op(old, x))
end

# `old => op(old, x)`, where `op(old, x)` is computed in Julia for an `atomicrmw`
@generated llvm_modify!(
    ptr::AnyPtr{T}, op, x::T, order::Val{o}, scope::Val, volatile::Val, align::Val, md::Val,
) where {T,o} = generate(modify_ir, T, op, o, false)

# only `old`, without computing `op(old, x)` for an `atomicrmw`
@generated llvm_fetch_modify!(
    ptr::AnyPtr{T}, op, x::T, order::Val{o}, scope::Val, volatile::Val, align::Val, md::Val,
) where {T,o} = generate(modify_ir, T, op, o, true)

# Compile the generators, which use LLVM.jl, as part of the package image.
let P = LLVMPtr{Int32,1}, F = LLVMPtr{Float32,1}
    md = ((:mmra, ((:a, :b),)),)
    fence_ir(:seq_cst, :device, md)
    load_ir(P, Int32, :monotonic, :device, false, 4, md)
    store_ir(P, Int32, :monotonic, :device, false, 4, md)
    rmw_ir(P, Int32, :add, :monotonic, :device, false, 4, md)
    cmpxchg_ir(P, Int32, :monotonic, :monotonic, :device, false, false, 4, md)
    cmpxchg_ir(F, Float32, :monotonic, :monotonic, :system, false, false, 4, ())
end
