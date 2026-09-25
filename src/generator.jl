# Atomic instructions as `llvmcall`s.
#
# Each primitive below compiles to exactly one LLVM atomic instruction, from IR that is
# generated for the pointer type, value type, ordering, syncscope, flags and metadata. All
# of those are type parameters, so these primitives don't depend on constant propagation;
# they are the layer that back-ends can rely on:
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

# Before Julia 1.12, llvmcall passes `Ptr` as an integer and uses typed pointers.
const OPAQUE_POINTERS = VERSION >= v"1.12-"

const AnyPtr{T} = Union{Ptr{T},LLVMPtr{T}}

address_space(::Type{<:Ptr}) = 0
address_space(::Type{<:LLVMPtr{<:Any,A}}) where {A} = A

llvm_pointer(pointee, A) =
    if OPAQUE_POINTERS
        A == 0 ? "ptr" : "ptr addrspace($A)"
    else
        A == 0 ? "$pointee*" : "$pointee addrspace($A)*"
    end

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

# The LLVM type of an atomic value of type `T`, or `nothing` if it isn't supported.
function llvm_type(::Type{T}) where {T}
    T === Float16 && return "half"
    T === Float32 && return "float"
    T === Float64 && return "double"
    HAS_BFLOAT16 && T === Core.BFloat16 && return "bfloat"
    T <: Ptr && return OPAQUE_POINTERS ? "ptr" : "i$WORD_SIZE"
    T <: LLVMPtr && return llvm_pointer("i8", address_space(T))
    isprimitivetype(T) && sizeof(T) in ATOMIC_SIZES && return "i$(8 * sizeof(T))"
    return nothing
end

is_zero_size(T) = Base.issingletontype(T)

# The IR that turns the pointer argument `%0` into a pointer to `lt`, the type of that
# pointer, and its value.
function pointer_argument(::Type{P}, lt) where {P}
    A = address_space(P)
    pt = llvm_pointer(lt, A)
    if OPAQUE_POINTERS
        return "", pt, "%0"
    elseif P <: Ptr
        return "%p = inttoptr i$WORD_SIZE %0 to $pt", pt, "%p"
    else
        return "%p = bitcast $(llvm_pointer("i8", A)) %0 to $pt", pt, "%p"
    end
end

normalize_order(o) = o === :acquire_release ? :acq_rel : o === :sequentially_consistent ? :seq_cst : o
julia_order(o) = o === :acq_rel ? :acquire_release : o === :seq_cst ? :sequentially_consistent : o
order_strength(o) = findfirst(==(o), (:unordered, :monotonic, :acquire, :release, :acq_rel, :seq_cst))

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

const RMW_OPERATIONS = (
    :xchg, :add, :sub, :and, :nand, :or, :xor, :max, :min, :umax, :umin,
    :fadd, :fsub, :fmax, :fmin,
    (Base.libllvm_version >= v"16" ? (:uinc_wrap, :udec_wrap) : ())...,
    (Base.libllvm_version >= v"20" ? (:usub_cond, :usub_sat) : ())...,
    (Base.libllvm_version >= v"21" ? (:fmaximum, :fminimum) : ())...,
)

const FLOAT_RMW_OPERATIONS = (:fadd, :fsub, :fmax, :fmin, :fmaximum, :fminimum)

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
    llvm_type(T) === nothing && invalid("unsupported atomic type ", T)
    align isa Integer && ispow2(align) && align >= sizeof(T) ||
        invalid("invalid alignment ", repr(align), " for an atomic ", T,
                ": expected a power of two, at least ", sizeof(T))
    return
end

function syncscope_string(scope)
    scope isa Symbol || invalid("invalid syncscope ", repr(scope))
    scope === :system && return ""
    return " syncscope(\"$(llvm_string(scope))\")"
end

function metadata_string(md)
    md isa Tuple || invalid("invalid metadata ", repr(md))
    io = IOBuffer()
    for entry in md
        entry isa Tuple{Symbol,Tuple} && entry[1] === :mmra ||
            invalid("unsupported metadata ", repr(entry))
        tags = entry[2]
        all(tag -> tag isa Tuple{Symbol,Symbol}, tags) || invalid("invalid MMRA tags ", repr(tags))
        isempty(tags) && continue
        nodes = ["!{!\"$(llvm_string(prefix))\", !\"$(llvm_string(suffix))\"}"
                 for (prefix, suffix) in tags]
        print(io, ", !mmra ", length(nodes) == 1 ? only(nodes) : "!{$(join(nodes, ", "))}")
    end
    return String(take!(io))
end

volatile_string(volatile) = volatile ? " volatile" : ""

# An atomic operation on a zero-size value only has to order memory.
fence_for(o, scope, md) =
    o === :unordered || o === :monotonic ? :(nothing) :
    :(llvm_fence(Val($(QuoteNode(o))), Val($(QuoteNode(scope))), Val($md)))

function fence_ir(order, scope, md)
    order = normalize_order(order)
    # like `Core.Intrinsics.atomic_fence`; LLVM would reject it
    order === :monotonic && return :(nothing)
    o = check_order(order, FENCE_ORDERS)
    ir = """
        fence$(syncscope_string(scope)) $o$(metadata_string(md))
        ret void
        """
    return :(llvmcall($ir, Cvoid, Tuple{}))
end

function load_ir(P, T, order, scope, volatile, align, md)
    o = check_order(order, LOAD_ORDERS)
    check_access(T, align, volatile)
    S, MD = syncscope_string(scope), metadata_string(md)
    is_zero_size(T) && return :($(fence_for(o, scope, md)); $(T.instance))
    use_intrinsics(P, T, scope, volatile, align, md) &&
        return :(Core.Intrinsics.atomic_pointerref(ptr, $(QuoteNode(julia_order(o)))))
    lt = llvm_type(T)
    setup, pt, p = pointer_argument(P, lt)
    ir = """
        $setup
        %v = load atomic$(volatile_string(volatile)) $lt, $pt $p$S $o, align $align$MD
        ret $lt %v
        """
    return :(llvmcall($ir, $T, Tuple{$P}, ptr))
end

function store_ir(P, T, order, scope, volatile, align, md)
    o = check_order(order, STORE_ORDERS)
    check_access(T, align, volatile)
    S, MD = syncscope_string(scope), metadata_string(md)
    is_zero_size(T) && return :($(fence_for(o, scope, md)); nothing)
    use_intrinsics(P, T, scope, volatile, align, md) &&
        return :(Core.Intrinsics.atomic_pointerset(ptr, x, $(QuoteNode(julia_order(o)))); nothing)
    lt = llvm_type(T)
    setup, pt, p = pointer_argument(P, lt)
    ir = """
        $setup
        store atomic$(volatile_string(volatile)) $lt %1, $pt $p$S $o, align $align$MD
        ret void
        """
    return :(llvmcall($ir, Cvoid, Tuple{$P,$T}, ptr, x))
end

function rmw_ir(P, T, op, order, scope, volatile, align, md)
    o = check_order(order, RMW_ORDERS)
    check_access(T, align, volatile)
    op in RMW_OPERATIONS ||
        invalid("unsupported atomicrmw operation ", repr(op), " on LLVM ", Base.libllvm_version)
    S, MD = syncscope_string(scope), metadata_string(md)
    is_zero_size(T) && return :($(fence_for(o, scope, md)); $(T.instance))
    if op in FLOAT_RMW_OPERATIONS ? !is_ieee_float(T) :
       op !== :xchg && (is_ieee_float(T) || T <: Ptr || T <: LLVMPtr)
        invalid("atomicrmw ", op, " doesn't apply to values of type ", T)
    end
    lt = llvm_type(T)
    setup, pt, p = pointer_argument(P, lt)
    ir = """
        $setup
        %v = atomicrmw$(volatile_string(volatile)) $op $pt $p, $lt %1$S $o, align $align$MD
        ret $lt %v
        """
    return :(llvmcall($ir, $T, Tuple{$P,$T}, ptr, x))
end

function cmpxchg_ir(P, T, success_order, failure_order, scope, weak, volatile, align, md)
    so = check_order(success_order, RMW_ORDERS)
    fo = check_order(failure_order, CAS_FAILURE_ORDERS)
    check_access(T, align, volatile, weak)
    S, MD = syncscope_string(scope), metadata_string(md)
    is_zero_size(T) &&
        return :($(fence_for(so, scope, md)); (old = $(T.instance), success = true))
    # the intrinsic is strong, and rejects a failure ordering stronger than the success one
    if !weak && use_intrinsics(P, T, scope, volatile, align, md) &&
       order_strength(fo) <= order_strength(so)
        return :(Core.Intrinsics.atomic_pointerreplace(
            ptr, cmp, new, $(QuoteNode(julia_order(so))), $(QuoteNode(julia_order(fo)))))
    end
    lt = llvm_type(T)
    # cmpxchg only takes integers and pointers
    ct = is_ieee_float(T) ? "i$(8 * sizeof(T))" : lt
    setup, pt, p = pointer_argument(P, ct)
    c, n, o = ct == lt ? ("%1", "%2", "%o") : ("%c", "%n", "%of")
    # llvmcall lowers a tuple of two i8s to an array
    rt = lt == "i8" ? "[2 x i8]" : "{ $lt, i8 }"
    ir = """
        $setup
        $(ct == lt ? "" : "%c = bitcast $lt %1 to $ct\n%n = bitcast $lt %2 to $ct")
        %r = cmpxchg$(weak ? " weak" : "")$(volatile_string(volatile)) $pt $p, $ct $c, $ct $n$S $so $fo, align $align$MD
        %o = extractvalue { $ct, i1 } %r, 0
        $(ct == lt ? "" : "%of = bitcast $ct %o to $lt")
        %s = extractvalue { $ct, i1 } %r, 1
        %s8 = zext i1 %s to i8
        %t = insertvalue $rt undef, $lt $o, 0
        %u = insertvalue $rt %t, i8 %s8, 1
        ret $rt %u
        """
    return quote
        old, success = llvmcall($ir, Tuple{$T,Bool}, Tuple{$P,$T,$T}, ptr, cmp, new)
        return (; old, success)
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
    llvm_type(T) === nothing && return nothing
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
               float && :fmaximum in RMW_OPERATIONS ? :fmaximum : nothing
    elseif op === min
        return T <: Base.BitSigned ? :min : T <: Base.BitUnsigned || bool ? :umin :
               float && :fminimum in RMW_OPERATIONS ? :fminimum : nothing
    elseif op === UnsafeAtomics.fmax
        return float ? :fmax : nothing
    elseif op === UnsafeAtomics.fmin
        return float ? :fmin : nothing
    end
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
