const ARGUMENTS_DOC = """
`ptr` is a `Ptr{T}` or a `Core.LLVMPtr{T,AS}` in any address space `AS`, and `T` a primitive
type of 1, 2, 4 or 8 bytes (and 16 on most 64-bit platforms), e.g. an integer, `Bool`, a
floating-point number or a pointer, or a type without fields.

`order` is an `UnsafeAtomics.Ordering`, e.g. `UnsafeAtomics.acquire`, or a `Symbol`, with
LLVM's or Julia's name, e.g. `:acq_rel` or `:acquire_release`. `scope` is an
`UnsafeAtomics.SyncScope`, or one of `:singlethread`, `:subgroup`, `:workgroup`, `:device`
and `:system` (see `UnsafeAtomics.default_scope`). Both can be run-time values: each
possible value then is a branch with its own instruction, without dynamic dispatch. Invalid
orderings throw a `ConcurrencyViolationError`. In GPU code, run-time values have to be a
`Union` of at most three orderings or scopes (see the README).

`volatile = true` makes the instruction volatile, and `align` sets its alignment, which has
to be a power of two, at least `sizeof(T)`, and a constant.
"""

@doc """
    UnsafeAtomics.load(ptr, order = seq_cst, scope = default_scope(ptr);
                       volatile = false, align = sizeof(T)) -> T

Load a value from `ptr` atomically, with a `load atomic` instruction. `order` can be
`unordered`, `monotonic`, `acquire` or `seq_cst`.

$ARGUMENTS_DOC
""" UnsafeAtomics.load

@doc """
    UnsafeAtomics.store!(ptr, x::T, order = seq_cst, scope = default_scope(ptr);
                         volatile = false, align = sizeof(T))

Store `x` to `ptr` atomically, with a `store atomic` instruction. `order` can be `unordered`,
`monotonic`, `release` or `seq_cst`.

$ARGUMENTS_DOC
""" UnsafeAtomics.store!

@doc """
    UnsafeAtomics.cas!(ptr, cmp::T, new::T, success_order = seq_cst,
                       failure_order = failure_order(success_order), scope = default_scope(ptr);
                       weak = false, volatile = false, align = sizeof(T))
        -> (; old::T, success::Bool)

Compare the value at `ptr` with `cmp` and, if they are equal, replace it with `new`, atomically,
with a `cmpxchg` instruction. Returns the value that was at `ptr`, and whether it was replaced.
The comparison is bitwise, also for floating-point numbers. With `weak = true`, the exchange
may fail spuriously.

`success_order` can be any ordering but `unordered`, and `failure_order`, used when the values
differ, `monotonic`, `acquire` or `seq_cst` (see `UnsafeAtomics.failure_order`).

$ARGUMENTS_DOC
""" UnsafeAtomics.cas!

@doc """
    UnsafeAtomics.modify!(ptr, op, x::T, order = seq_cst, scope = default_scope(ptr);
                          volatile = false, align = sizeof(T)) -> Pair{T,T}

Replace the value `old` at `ptr` with `op(old, x)` atomically, and return `old => op(old, x)`.
`order` can be any ordering but `unordered`.

Where an `atomicrmw` instruction implements `op` for `T`, `modify!` is that one instruction,
and `op(old, x)` in the result is computed by Julia, after the instruction: `op` has to be
defined for `T` (unlike for the named functions like `add!`, which only return `old`). For
floating-point numbers, it can differ from the stored value in NaN payloads or, for
`fmax`/`fmin`, in the sign of zero. These are:

| `op` | `T` | `atomicrmw` |
|:-----|:----|:------------|
| `+`, `-` | integers; `Float16`, `Float32`, `Float64`, and `Core.BFloat16` from LLVM 20 | `add`, `sub`; `fadd`, `fsub` |
| `&`, `\\|`, `xor` | integers, `Bool` | `and`, `or`, `xor` |
| `⊼` | integers | `nand` |
| `max`, `min` | integers, `Bool` | `max`, `min`, `umax`, `umin` |
| `max`, `min` | floating-point numbers, from LLVM 21 | `fmaximum`, `fminimum` |
| `UnsafeAtomics.fmax`, `UnsafeAtomics.fmin` | floating-point numbers (`Core.BFloat16` from LLVM 20) | `fmax`, `fmin` |
| `UnsafeAtomics.right` | any | `xchg` |
| `UnsafeAtomics.inc_wrap`, `dec_wrap`, `sub_cond`, `sub_sat` | unsigned integers, from LLVM 22 | `uinc_wrap`, `udec_wrap`, `usub_cond`, `usub_sat` |

Other operations, e.g. on older versions of LLVM, use a loop of `load` and weak `cas!`, which
computes `op(old, x)` in Julia and stores exactly that.

$ARGUMENTS_DOC
""" UnsafeAtomics.modify!

for (op, name) in OP_RMW_TABLE
    fn = Symbol(name, "!")
    opname = parentmodule(op) === UnsafeAtomics ? "UnsafeAtomics.$(nameof(op))" : string(op)
    doc = """
        UnsafeAtomics.$fn(ptr, x::T, order = seq_cst, scope = default_scope(ptr);
                          volatile = false, align = sizeof(T)) -> T

    Like `first(modify!(ptr, $opname, x, ...))`: replace the value at `ptr` atomically, and
    return the old one, without computing the new one in Julia. See `UnsafeAtomics.modify!`.
    """
    @eval @doc $doc UnsafeAtomics.$fn
end

@doc """
    UnsafeAtomics.fence(order = seq_cst, scope = system)

Order memory accesses, with a `fence` instruction. `order` can be `acquire`, `release`,
`acq_rel` or `seq_cst`; `monotonic` does nothing, and `unordered` throws a
`ConcurrencyViolationError`. See `UnsafeAtomics.load` for the arguments.
""" UnsafeAtomics.fence

@doc """
    UnsafeAtomics.right(old, x) = x

The operation of `modify!` that replaces the value, with `atomicrmw xchg`.
""" UnsafeAtomics.right

@doc """
    UnsafeAtomics.Ordering

The type of the atomic orderings, which are LLVM's: `UnsafeAtomics.unordered`, `monotonic`,
`acquire`, `release`, `acq_rel` and `seq_cst`, also available with Julia's names as
`acquire_release` and `sequentially_consistent`.
""" UnsafeAtomics.Ordering

@doc """
    UnsafeAtomics.SyncScope

The type of the synchronization scopes, which are LLVM's `syncscope`s: the set of threads an
atomic operation synchronizes with. `UnsafeAtomics.singlethread`, `subgroup`, `workgroup`,
`device` and `system` (also called `none`) are the canonical ones, which the GPU back-ends map
to their targets' scopes. `UnsafeAtomics.SyncScope(name)` makes others.
""" UnsafeAtomics.SyncScope
