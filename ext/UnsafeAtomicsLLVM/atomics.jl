using LLVM, LLVM.IR, LLVM.Build
using LLVM.Interop

const MEMORY_ORDERING_EXPLANATION = """
specified as a symbol (e.g., `:sequentially_consistent`) or a `Val` of a symbol (e.g.,
`Val(:sequentially_consistent)`)
"""

"""
    atomic_pointerref(pointer::LLVMPtr{T}, ordering) -> value::T
Load a `value` from `pointer` with the given memory `ordering` atomically.
`ordering` is a Julia atomic ordering $MEMORY_ORDERING_EXPLANATION.
See also: `getproperty`, `getfield`
"""
atomic_pointerref

"""
    LLVM.Interop.atomic_pointerset(pointer::LLVMPtr{T}, x::T, ordering) -> pointer
Store a value `x` in `pointer` with the given memory `ordering` atomically.
`ordering` is a Julia atomic ordering $MEMORY_ORDERING_EXPLANATION.
See also: `setproperty!`, `setfield!`
"""
atomic_pointerset

"""
    atomic_pointermodify(
        pointer::LLVMPtr{T},
        op,
        x::T,
        ordering,
    ) -> (old => new)::Pair{T,T}
Replace an `old` value stored at `pointer` with a `new` value comped as `new = op(old, x)`
with the given memory `ordering` atomically.  Return a pair `old => new`.
`ordering` is a Julia atomic ordering $MEMORY_ORDERING_EXPLANATION.
See also: `modifyproperty!`, `modifyfield!`
"""
atomic_pointermodify

"""
    atomic_pointerswap(pointer::LLVMPtr{T}, op, new::T, ordering) -> old::T
Replace an `old` value stored at `pointer` with a `new` value with the given memory
`ordering` atomically.  Return the `old` value.
`ordering` is a Julia atomic ordering $MEMORY_ORDERING_EXPLANATION.
See also: `modifyproperty!`, `modifyfield!`
"""
atomic_pointerswap

"""
    atomic_pointerreplace(
        pointer::LLVMPtr{T},
        expected::T,
        desired::T,
        success_ordering,
        fail_ordering,
    ) -> (; old::T, success::Bool)
Try to replace the value of `pointer` from an `expected` value to a `desired` value
atomically with the ordering `success_ordering`.  The property `old` of the returned value
is the value stored in the `pointer`.  The property `success` of the returned value
indicates if the replacement was successful.  The ordering `fail_ordering` specifies the
ordering used for loading the `old` value.
`success_ordering` and `fail_ordering` are Julia atomic orderings
$MEMORY_ORDERING_EXPLANATION.
See also: `replaceproperty!`, `replacefield!`
"""
atomic_pointerreplace

const _llvm_from_julia_ordering = (
    not_atomic = LLVM.API.LLVMAtomicOrderingNotAtomic,
    unordered = LLVM.API.LLVMAtomicOrderingUnordered,
    monotonic = LLVM.API.LLVMAtomicOrderingMonotonic,
    acquire = LLVM.API.LLVMAtomicOrderingAcquire,
    release = LLVM.API.LLVMAtomicOrderingRelease,
    acquire_release = LLVM.API.LLVMAtomicOrderingAcquireRelease,
    sequentially_consistent = LLVM.API.LLVMAtomicOrderingSequentiallyConsistent,
)

_julia_ordering(p) =
    Union{map(x -> p(x) ? Val{x} : Union{}, keys(_llvm_from_julia_ordering))...}

const AllOrdering = _julia_ordering(_ -> true)
const AtomicOrdering = _julia_ordering(!=(:not_atomic))

const LLVMOrderingVal = Union{map(x -> Val{x}, values(_llvm_from_julia_ordering))...}

is_stronger_than_monotonic(order::Symbol) =
    !(order === :monotonic || order === :unordered || order === :not_atomic)

for (julia, llvm) in pairs(_llvm_from_julia_ordering)
    @eval llvm_from_julia_ordering(::Val{$(QuoteNode(julia))}) = Val{$llvm}()
end

_valueof(::Val{x}) where {x} = x

@inline function atomic_pointerref(ptr::LLVMPtr{T}, order::AllOrdering, sync) where {T}
    sizeof(T) == 0 && return T.instance
    return llvm_atomic_load(ptr, llvm_from_julia_ordering(order), sync)
end

# Non-atomic accesses don't have a synchronization scope.
access_scope(order, scope) =
    order == LLVM.API.LLVMAtomicOrderingNotAtomic ? nothing : String(scope)

@llvmgenerated builder function llvm_atomic_load(
    ptr::LLVMPtr{T,A},
    ::Val{order},
    ::Val{sync},
)::T where {T,A,order,sync}
    eltyp = convert(LLVMType, T)
    typed_ptr = bitcast!(builder, ptr, LLVM.PointerType(eltyp, A))
    ld = load!(builder, eltyp, typed_ptr; ordering = order,
               scope = access_scope(order, sync), align = sizeof(T))
    if A != 0
        ld.metadata[MD_tbaa] = tbaa_addrspace(A)
    end
    return ld
end

@generated function atomic_pointerset(
    ptr::LLVMPtr{T,A},
    x::T,
    order::AllOrdering,
    sync,
) where {T,A}
    if sizeof(T) == 0
        # Mimicking what `Core.Intrinsics.atomic_pointerset` generates.
        # See: https://github.com/JuliaLang/julia/blob/v1.7.2/src/cgutils.cpp#L1570-L1572
        if VERSION < v"1.14.0-DEV.1371"
            return quote
                is_stronger_than_monotonic(_valueof(order)) || return ptr
                Core.Intrinsics.atomic_fence(_valueof(order))
                return ptr
            end
        else
            return quote
                is_stronger_than_monotonic(_valueof(order)) || return ptr
                Core.Intrinsics.atomic_fence(_valueof(order), :system)
                return ptr
            end
        end
    end
    quote
        llvm_atomic_store(ptr, x, llvm_from_julia_ordering(order), sync)
        ptr
    end
end

@llvmgenerated builder function llvm_atomic_store(
    ptr::LLVMPtr{T,A},
    x::T,
    ::Val{order},
    ::Val{sync},
)::Nothing where {T,A,order,sync}
    typed_ptr = bitcast!(builder, ptr, LLVM.PointerType(convert(LLVMType, T), A))
    st = store!(builder, x, typed_ptr; ordering = order,
                scope = access_scope(order, sync), align = sizeof(T))
    if A != 0
        st.metadata[MD_tbaa] = tbaa_addrspace(A)
    end
    return nothing
end

right(_, r) = r

const binoptable = [
    (:xchg, right, LLVM.API.LLVMAtomicRMWBinOpXchg),
    (:add, +, LLVM.API.LLVMAtomicRMWBinOpAdd),
    (:sub, -, LLVM.API.LLVMAtomicRMWBinOpSub),
    (:and, &, LLVM.API.LLVMAtomicRMWBinOpAnd),
    (:or, |, LLVM.API.LLVMAtomicRMWBinOpOr),
    (:xor, xor, LLVM.API.LLVMAtomicRMWBinOpXor),
    (:max, max, LLVM.API.LLVMAtomicRMWBinOpMax),
    (:min, min, LLVM.API.LLVMAtomicRMWBinOpMin),
    (:umax, max, LLVM.API.LLVMAtomicRMWBinOpUMax),
    (:umin, min, LLVM.API.LLVMAtomicRMWBinOpUMin),
    (:fadd, +, LLVM.API.LLVMAtomicRMWBinOpFAdd),
    (:fsub, -, LLVM.API.LLVMAtomicRMWBinOpFSub),
    (:fmax, max, LLVM.API.LLVMAtomicRMWBinOpFMax),
    (:fmin, min, LLVM.API.LLVMAtomicRMWBinOpFMin),
]

const AtomicRMWBinOpVal = Union{(Val{binop} for (_, _, binop) in binoptable)...}

@llvmgenerated builder function llvm_atomic_op(
    binop::AtomicRMWBinOpVal,
    ptr::LLVMPtr{T,A},
    val::T,
    order::LLVMOrderingVal,
    ::Val{sync},
)::T where {T,A,sync}
    typed_ptr = bitcast!(builder, ptr, LLVM.PointerType(convert(LLVMType, T), A))
    return atomic_rmw!(builder, _valueof(binop), typed_ptr, val, _valueof(order);
                       scope = String(sync))
end

@inline function atomic_pointermodify(
    ptr::LLVMPtr{T},
    ::typeof(right),
    x::T,
    order::AtomicOrdering,
    sync::Val{S}
) where {T, S}
    old = llvm_atomic_op(
        Val(LLVM.API.LLVMAtomicRMWBinOpXchg),
        ptr,
        x,
        llvm_from_julia_ordering(order),
        sync
    )
    return old => x
end

const atomictypes = Any[
    Int8,
    Int16,
    Int32,
    Int64,
    Int128,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    UInt128,
    Float16,
    Float32,
    Float64,
]

for (opname, op, llvmop) in binoptable
    opname === :xchg && continue
    types = if opname in (:min, :max)
        filter(t -> t <: Signed, atomictypes)
    elseif opname in (:umin, :umax)
        filter(t -> t <: Unsigned, atomictypes)
    elseif opname in (:fadd, :fsub, :fmin, :fmax)
        filter(t -> t <: AbstractFloat, atomictypes)
    else
        filter(t -> t <: Integer, atomictypes)
    end
    for T in types
        @eval @inline function atomic_pointermodify(
            ptr::LLVMPtr{$T},
            ::$(typeof(op)),
            x::$T,
            order::AtomicOrdering,
            sync::Val{S},
        ) where {S}
            old = llvm_atomic_op(
                $(Val(llvmop)), ptr, x, llvm_from_julia_ordering(order), sync)
            return old => $op(old, x)
        end
    end
end

# @inline atomic_pointerswap(pointer, new) = first(atomic_pointermodify(pointer, right, new))
@inline atomic_pointerswap(pointer, new, order, sync) =
    first(atomic_pointermodify(pointer, right, new, order, sync))

@inline function atomic_pointermodify(
    ptr::LLVMPtr{T},
    op,
    x::T,
    order::AllOrdering,
    sync::S,
) where {T, S}
    # Should `fail_order` be stronger?  Ref: https://github.com/JuliaLang/julia/issues/45256
    fail_order = Val(:monotonic)
    old = atomic_pointerref(ptr, fail_order, sync)
    while true
        new = op(old, x)
        (old, success) = atomic_pointerreplace(ptr, old, new, order, fail_order, sync)
        success && return old => new
    end
end

@inline function llvm_atomic_cas(
    ptr::LLVMPtr{T},
    cmp::T,
    val::T,
    success_order::LLVMOrderingVal,
    fail_order::LLVMOrderingVal,
    sync,
) where {T}
    success = Ref{Int8}()
    old = GC.@preserve success begin
        success_ptr = Ptr{Int8}(pointer_from_objref(success))
        _llvm_atomic_cas(ptr, cmp, val, success_order, fail_order, sync, success_ptr)
    end
    (; old, success = success[] != zero(Int8))
end

@llvmgenerated builder function _llvm_atomic_cas(
    ptr::LLVMPtr{T,A},
    cmp::T,
    val::T,
    ::Val{success_order},
    ::Val{fail_order},
    ::Val{sync},
    success_ptr::Ptr{Int8},
)::T where {T,A,success_order,fail_order,sync}
    T_val = convert(LLVMType, T)
    T_pointee = T_val
    if T_val isa LLVM.FloatingPointType
        T_pointee = LLVM.IntType(sizeof(T) * 8)
    end

    typed_ptr = bitcast!(builder, ptr, LLVM.PointerType(T_pointee, A))
    # before Julia 1.12, `llvmcall` passes a `Ptr` as an integer
    T_ok_ptr = LLVM.PointerType(LLVM.Int8Type())
    ok_ptr = success_ptr.value_type isa LLVM.IntegerType ?
             inttoptr!(builder, success_ptr, T_ok_ptr) : success_ptr

    cmp_int = cmp
    if T_val isa LLVM.FloatingPointType
        cmp_int = bitcast!(builder, cmp_int, T_pointee)
    end

    val_int = val
    if T_val isa LLVM.FloatingPointType
        val_int = bitcast!(builder, val_int, T_pointee)
    end

    res = atomic_cmpxchg!(
        builder,
        typed_ptr,
        cmp_int,
        val_int,
        success_order,
        fail_order;
        scope = String(sync),
    )

    rv = extract_value!(builder, res, 0)
    ok = extract_value!(builder, res, 1)
    ok = zext!(builder, ok, LLVM.Int8Type())
    store!(builder, ok, ok_ptr)

    if T_val isa LLVM.FloatingPointType
        rv = bitcast!(builder, rv, T_val)
    end

    return rv
end

@inline function atomic_pointerreplace(
    ptr::LLVMPtr{T},
    expected::T,
    desired::T,
    ::Val{:not_atomic},
    ::Val{:not_atomic},
    sync,
) where {T}
    old = atomic_pointerref(ptr, Val(:not_atomic), sync)
    if old === expected
        atomic_pointerset(ptr, desired, Val(:not_atomic), sync)
        success = true
    else
        success = false
    end
    return (; old, success)
end

@inline atomic_pointerreplace(
    ptr::LLVMPtr{T},
    expected::T,
    desired::T,
    success_order::_julia_ordering(∉((:not_atomic, :unordered))),
    fail_order::_julia_ordering(∉((:not_atomic, :unordered, :release, :acquire_release))),
    sync
) where {T} = llvm_atomic_cas(
    ptr,
    expected,
    desired,
    llvm_from_julia_ordering(success_order),
    llvm_from_julia_ordering(fail_order),
    sync
)
