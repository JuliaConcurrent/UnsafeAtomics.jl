@inline UnsafeAtomics.load(x) = UnsafeAtomics.load(x, seq_cst)
@inline UnsafeAtomics.store!(x, v) = UnsafeAtomics.store!(x, v, seq_cst)
@inline UnsafeAtomics.cas!(x, cmp, new) = UnsafeAtomics.cas!(x, cmp, new, seq_cst, seq_cst)
@inline UnsafeAtomics.modify!(ptr, op, x) = UnsafeAtomics.modify!(ptr, op, x, seq_cst)
@inline UnsafeAtomics.fence() = UnsafeAtomics.fence(seq_cst)

@inline UnsafeAtomics.load(x, ord) = UnsafeAtomics.load(x, ord, none)
@inline UnsafeAtomics.store!(x, v, ord) = UnsafeAtomics.store!(x, v, ord, none)
@inline UnsafeAtomics.cas!(x, cmp, new, ord) = UnsafeAtomics.cas!(x, cmp, new, ord, failure_order(ord), none)
@inline UnsafeAtomics.cas!(x, cmp, new, success_ord, failure_ord) = UnsafeAtomics.cas!(x, cmp, new, success_ord, failure_ord, none)
@inline UnsafeAtomics.modify!(ptr, op, x, ord) = UnsafeAtomics.modify!(ptr, op, x, ord, none)
@inline UnsafeAtomics.fence(ord) = UnsafeAtomics.fence(ord, none)

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

for (op, rmwop) in OP_RMW_TABLE
    fn = Symbol(rmwop, "!")
    @eval @inline UnsafeAtomics.$fn(x, v) = UnsafeAtomics.$fn(x, v, seq_cst)
    @eval @inline UnsafeAtomics.$fn(x, v, ord) = UnsafeAtomics.$fn(x, v, ord, none) 
    @eval @inline UnsafeAtomics.$fn(ptr, x, ord, scope) =
        first(UnsafeAtomics.modify!(ptr, $op, x, ord, scope))
end

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

@noinline throw_invalid_ordering() =
    throw(Base.ConcurrencyViolationError("invalid atomic ordering"))

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
        function UnsafeAtomics.fence(ord::Ordering, ::typeof(none))
            Core.Intrinsics.atomic_fence(base_ordering(ord))
            return nothing
        end
    else
        function UnsafeAtomics.fence(ord::Ordering, ::typeof(none))
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
            @eval UnsafeAtomics.fence(::$(typeof(ord)), ::typeof(none)) = nothing
        elseif ord === unordered
            # defined below
        elseif ord === seq_cst && X86_FENCE_WORKAROUND
            # defined by the x86_64 special case below
        else
            @eval function UnsafeAtomics.fence(::$(typeof(ord)), ::typeof(none))
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
UnsafeAtomics.fence(::typeof(unordered), ::typeof(none)) = throw_invalid_ordering()
if X86_FENCE_WORKAROUND
    UnsafeAtomics.fence(::typeof(seq_cst), ::typeof(none)) = cpu_seq_cst_fence()
end

# Fences in other scopes. Like for the system scope, `monotonic` is a no-op and `unordered`
# is invalid, as `fence` requires at least `acquire`.
UnsafeAtomics.fence(ord::Ordering, sync::LLVMSyncScope) =
    llvm_fence(Val(llvm_ordering(ord)), Val(scope_name(sync)), Val(()))

# Pointers. The generator emits one instruction for each, or uses the intrinsics for Ptr in
# the system scope.

@inline UnsafeAtomics.load(ptr::AnyPtr{T}, order::Ordering, scope::LLVMSyncScope) where {T} =
    llvm_load(ptr, Val(llvm_ordering(order)), Val(scope_name(scope)), Val(false),
              Val(sizeof(T)), Val(()))

@inline function UnsafeAtomics.store!(
    ptr::AnyPtr{T},
    x::T,
    order::Ordering,
    scope::LLVMSyncScope,
) where {T}
    llvm_store!(ptr, x, Val(llvm_ordering(order)), Val(scope_name(scope)), Val(false),
                Val(sizeof(T)), Val(()))
end

@inline UnsafeAtomics.cas!(
    ptr::AnyPtr{T},
    cmp::T,
    new::T,
    success_ordering::Ordering,
    failure_ordering::Ordering,
    scope::LLVMSyncScope,
) where {T} = llvm_cmpxchg!(
    ptr,
    cmp,
    new,
    Val(llvm_ordering(success_ordering)),
    Val(llvm_ordering(failure_ordering)),
    Val(scope_name(scope)),
    Val(false),
    Val(false),
    Val(sizeof(T)),
    Val(()),
)

@inline UnsafeAtomics.modify!(
    ptr::AnyPtr{T},
    op::OP,
    x::T,
    order::Ordering,
    scope::LLVMSyncScope,
) where {T,OP} = llvm_modify!(ptr, op, x, Val(llvm_ordering(order)), Val(scope_name(scope)),
                              Val(false), Val(sizeof(T)), Val(()))

for (op, rmwop) in OP_RMW_TABLE
    fn = Symbol(rmwop, "!")
    @eval @inline UnsafeAtomics.$fn(
        ptr::AnyPtr{T},
        x::T,
        order::Ordering,
        scope::LLVMSyncScope,
    ) where {T} = llvm_fetch_modify!(ptr, $op, x, Val(llvm_ordering(order)),
                                     Val(scope_name(scope)), Val(false), Val(sizeof(T)), Val(()))
end
