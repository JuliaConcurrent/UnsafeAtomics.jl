#! format: off
if 16 in ATOMIC_SIZES
    const inttypes = (Int8, Int16, Int32, Int64, Int128,
                      UInt8, UInt16, UInt32, UInt64, UInt128)
else
    const inttypes = (Int8, Int16, Int32, Int64,
                      UInt8, UInt16, UInt32, UInt64)
end
const floattypes = (Float16, Float32, Float64)
#! format: on

const OP_RMW_TABLE = [
    (+) => :add,
    (-) => :sub,
    right => :xchg,
    (&) => :and,
    (⊼) => :nand,
    (|) => :or,
    (⊻) => xor,
    max => :max,
    min => :min,
    UnsafeAtomics.fmax => :fmax,
    UnsafeAtomics.fmin => :fmin,
    UnsafeAtomics.inc_wrap => :inc_wrap,
    UnsafeAtomics.dec_wrap => :dec_wrap,
    UnsafeAtomics.sub_cond => :sub_cond,
    UnsafeAtomics.sub_sat => :sub_sat,
]

const FMAX_DOC = """
    UnsafeAtomics.fmax(x, y)
    UnsafeAtomics.fmin(x, y)

The maximum and minimum of floating-point numbers as defined by IEEE 754 `maxNum` and
`minNum`: unlike `max` and `min`, a NaN operand is ignored in favour of the other one.
`modify!` and `fmax!`/`fmin!` use the `atomicrmw fmax`/`fmin` instructions for these, whose
choice between zeros of opposite signs, and of NaN payloads, depends on the target and LLVM
version.
"""
@doc FMAX_DOC UnsafeAtomics.fmax(x::T, y::T) where {T<:AbstractFloat} =
    isnan(x) ? y : isnan(y) ? x : max(x, y)
@doc FMAX_DOC UnsafeAtomics.fmin(x::T, y::T) where {T<:AbstractFloat} =
    isnan(x) ? y : isnan(y) ? x : min(x, y)

"""
    UnsafeAtomics.inc_wrap(old, x)

`old + 1`, wrapping around to zero past `x`: `old >= x ? 0 : old + 1`, for unsigned
integers. `modify!` and `inc_wrap!` use `atomicrmw uinc_wrap` for it from LLVM 22 (Julia 1.14).
"""
UnsafeAtomics.inc_wrap(old::T, x::T) where {T<:Unsigned} = old >= x ? zero(T) : old + one(T)

"""
    UnsafeAtomics.dec_wrap(old, x)

`old - 1`, wrapping around to `x` at zero or above `x`: `(old == 0 || old > x) ? x : old - 1`,
for unsigned integers. `modify!` and `dec_wrap!` use `atomicrmw udec_wrap` for it from LLVM 22
(Julia 1.14).
"""
UnsafeAtomics.dec_wrap(old::T, x::T) where {T<:Unsigned} =
    (iszero(old) || old > x) ? x : old - one(T)

"""
    UnsafeAtomics.sub_cond(old, x)

`old - x` if that doesn't wrap around, `old` otherwise, for unsigned integers. `modify!` and
`sub_cond!` use `atomicrmw usub_cond` for it from LLVM 22 (Julia 1.14).
"""
UnsafeAtomics.sub_cond(old::T, x::T) where {T<:Unsigned} = old >= x ? old - x : old

"""
    UnsafeAtomics.sub_sat(old, x)

`old - x`, saturating at zero, for unsigned integers. `modify!` and `sub_sat!` use
`atomicrmw usub_sat` for it from LLVM 22 (Julia 1.14).
"""
UnsafeAtomics.sub_sat(old::T, x::T) where {T<:Unsigned} = old >= x ? old - x : zero(T)

# Before JuliaLang/julia#57806, inference modeled `Core.Intrinsics.atomic_fence` as
# effect-free, so a fence inlined through the public wrapper could be deleted when its
# result was unused. Probe the effect model directly instead of hard-coding versions, so
# this relaxes by itself on any patch release that carries the backport. If the compiler
# API is unavailable, retain the workaround.
const FENCE_INTRINSIC_ELIDABLE =
    VERSION < v"1.12" && try
        Core.Compiler.is_effect_free(
            Base.infer_effects(Core.Intrinsics.atomic_fence, Tuple{Symbol}),
        )
    catch
        true
    end

# Before LLVM 20, a seq_cst fence on x86_64 lowers to `mfence`, which is slow on AMD CPUs.
# Emit a locked `or` instead, like LLVM does since llvm/llvm-project#106555.
const X86_FENCE_WORKAROUND = Sys.ARCH == :x86_64 && Base.libllvm_version < v"20"

# The system-scope seq_cst fence on the host CPU. GPU back-ends overlay this function with a
# plain `fence seq_cst`, as the x86 assembly below must not end up in device code.
if X86_FENCE_WORKAROUND
    @inline cpu_seq_cst_fence() = Base.llvmcall(
        (raw"""
        define void @fence() #0 {
        entry:
            tail call void asm sideeffect "lock orq $$0 , (%rsp)", ""(); should this have ~{memory}
            ret void
        }
        attributes #0 = { alwaysinline }
        """, "fence"), Nothing, Tuple{})
else
    @inline cpu_seq_cst_fence() = llvmcall("""
        fence seq_cst
        ret void
        """, Cvoid, Tuple{})
end

if !FENCE_INTRINSIC_ELIDABLE
    # Core.Intrinsics.atomic_fence was introduced in 1.10
    if VERSION < v"1.14.0-DEV.1371"
        function system_fence(ord::Ordering)
            Core.Intrinsics.atomic_fence(base_ordering(ord))
            return nothing
        end
    else
        function system_fence(ord::Ordering)
            Core.Intrinsics.atomic_fence(base_ordering(ord), :system)
            return nothing
        end
    end
else
    # Inference treats `llvmcall` conservatively, so the fence is retained.
    for ord in orderings
        if ord === monotonic
            # `fence` requires at least `acquire`; the intrinsic accepts
            # `:monotonic` and codegen turns it into a no-op.
            @eval system_fence(::$(typeof(ord))) = nothing
        elseif ord === unordered
            # defined below
        elseif ord === seq_cst && X86_FENCE_WORKAROUND
            # defined by the x86_64 special case below
        else
            @eval function system_fence(::$(typeof(ord)))
                return llvmcall(
                    $("""
                    fence $ord
                    ret void
                    """),
                    Cvoid,
                    Tuple{},
                )
            end
        end
    end
end
# A fence can't be unordered. Throw the intrinsic's error ourselves: Julia's inference thinks
# the intrinsic throws another type of exception, which Julia 1.11 miscompiles when the error
# is caught. A call that always throws can't be elided.
system_fence(::typeof(unordered)) = throw_invalid_ordering()
if X86_FENCE_WORKAROUND
    system_fence(::typeof(seq_cst)) = cpu_seq_cst_fence()
end

@inline UnsafeAtomics.fence(order = seq_cst, scope = system) =
    with_ordering_and_scope(fence_in, order, scope)
@inline fence_in(::Val{o}, ::Val{:system}) where {o} = system_fence(LLVMOrdering{o}())
# Other scopes. Like for the system scope, `monotonic` is a no-op and `unordered` is invalid,
# as `fence` requires at least `acquire`.
@inline fence_in(o::Val, s::Val) = llvm_fence(o, s, Val(()))

# Pointers. The generator emits one instruction for each, or uses the intrinsics for Ptr in
# the system scope. Orderings and scopes are selected by value: see `with_ordering`.

@inline UnsafeAtomics.load(
    ptr::AnyPtr{T},
    order = seq_cst,
    scope = default_scope(ptr);
    volatile::Bool = false,
    align::Integer = sizeof(T),
) where {T} =
    with_ordering_and_scope(order, scope, flag(volatile), Val(align)) do o, s, v, al
        llvm_load(ptr, o, s, v, al, Val(()))
    end

@inline UnsafeAtomics.store!(
    ptr::AnyPtr{T},
    x::T,
    order = seq_cst,
    scope = default_scope(ptr);
    volatile::Bool = false,
    align::Integer = sizeof(T),
) where {T} =
    with_ordering_and_scope(order, scope, flag(volatile), Val(align)) do o, s, v, al
        llvm_store!(ptr, x, o, s, v, al, Val(()))
    end

@inline UnsafeAtomics.cas!(
    ptr::AnyPtr{T},
    cmp::T,
    new::T,
    success = seq_cst,
    failure = failure_order(success),
    scope = default_scope(ptr);
    weak::Bool = false,
    volatile::Bool = false,
    align::Integer = sizeof(T),
) where {T} =
    with_orderings_and_scope(success, failure, scope, flag(weak), flag(volatile), Val(align)) do so, fo, s, w, v, al
        llvm_cmpxchg!(ptr, cmp, new, so, fo, s, w, v, al, Val(()))
    end

@inline UnsafeAtomics.modify!(
    ptr::AnyPtr{T},
    op::OP,
    x::T,
    order = seq_cst,
    scope = default_scope(ptr);
    volatile::Bool = false,
    align::Integer = sizeof(T),
) where {T,OP} =
    with_ordering_and_scope(order, scope, flag(volatile), Val(align)) do o, s, v, al
        llvm_modify!(ptr, op, x, o, s, v, al, Val(()))
    end

for (op, rmwop) in OP_RMW_TABLE
    fn = Symbol(rmwop, "!")
    @eval @inline UnsafeAtomics.$fn(
        ptr::AnyPtr{T},
        x::T,
        order = seq_cst,
        scope = default_scope(ptr);
        volatile::Bool = false,
        align::Integer = sizeof(T),
    ) where {T} =
        with_ordering_and_scope(order, scope, flag(volatile), Val(align)) do o, s, v, al
            llvm_fetch_modify!(ptr, $op, x, o, s, v, al, Val(()))
        end
end
