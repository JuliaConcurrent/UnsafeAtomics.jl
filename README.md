# UnsafeAtomics

Atomic operations on raw pointers, with LLVM's orderings and synchronization scopes. Each
operation is one LLVM atomic instruction (`load atomic`, `store atomic`, `atomicrmw`, `cmpxchg`
or `fence`), except for read-modify-write operations that have no instruction, which are a loop
of `load atomic` and `cmpxchg`, and operations on values without fields, which only need a
`fence` (or nothing, for `unordered` and `monotonic`). That makes UnsafeAtomics the common front-end for atomics on
CPUs and GPUs: the GPU back-ends lower these instructions for their targets.

```julia
using UnsafeAtomics: UnsafeAtomics as UA

xs = zeros(Int32, 4)
GC.@preserve xs begin
    p = pointer(xs)
    UA.store!(p, Int32(1), UA.release)
    UA.load(p, UA.acquire)                       # 1
    UA.add!(p, Int32(2), UA.acq_rel)             # 1, the old value
    UA.modify!(p, *, Int32(3))                   # 3 => 9
    UA.cas!(p, Int32(9), Int32(0))               # (old = 9, success = true)
    UA.fence(UA.seq_cst)
end
```

Nothing is exported; the API is public on Julia 1.11 and later.

## Operations

| function | instruction |
|:---------|:------------|
| `load(ptr, order = seq_cst, scope = system)` | `load atomic` |
| `store!(ptr, x, order = seq_cst, scope = system)` | `store atomic` |
| `cas!(ptr, cmp, new, success = seq_cst, failure = failure_order(success), scope = system)` | `cmpxchg`, returns `(; old, success)` |
| `modify!(ptr, op, x, order = seq_cst, scope = system)` | `atomicrmw`, or a compare-and-swap loop; returns `old => op(old, x)` |
| `add!`, `sub!`, `xchg!`, `and!`, `nand!`, `or!`, `xor!`, `max!`, `min!`, `fmax!`, `fmin!`, `inc_wrap!`, `dec_wrap!`, `sub_cond!`, `sub_sat!` | `atomicrmw`, or a compare-and-swap loop; return the old value |
| `fence(order = seq_cst, scope = system)` | `fence` |

- **Pointers:** `Ptr{T}`, and `Core.LLVMPtr{T,AS}` in any address space.
- **Values:** `x`, `cmp` and `new` are of type `T`: a primitive type of 1, 2, 4 or 8 bytes (and
  16 on most 64-bit platforms), e.g. integers, `Bool`, `Float16`/`Float32`/`Float64`,
  `Core.BFloat16`, pointers, or a type without fields.
- **Keywords:** `volatile = true`, `align = n` (a constant power of two, at least `sizeof(T)`;
  ignored for types without fields), and `weak = true` for `cas!`.

## Orderings

`UA.unordered`, `monotonic`, `acquire`, `release`, `acq_rel` and `seq_cst` are LLVM's
orderings; `acquire_release` and `sequentially_consistent` are aliases with Julia's names. They
can also be passed as a `Symbol`, with either name. Not every ordering is valid for every
operation: `load` can't release, `store!` can't acquire, read-modify-write operations can't be
`unordered`, a `cas!` failure ordering can't release, and a `fence` needs at least `acquire`
(`monotonic` is a no-op). Invalid orderings throw a `ConcurrencyViolationError`.
UnsafeAtomics doesn't strengthen or weaken orderings.

## Synchronization scopes

The scope is the set of threads an operation synchronizes with. `UA.singlethread`,
`subgroup`, `workgroup`, `device` and `system` are the canonical scopes, which GPU back-ends
map to their targets' scopes; they can also be passed as a `Symbol`. `UA.SyncScope(name)` makes
other LLVM syncscopes, for scopes specific to a target. The default is `system`, as in LLVM, on
every pointer; GPU code usually wants `device`. On CPUs, all scopes but `singlethread` are the
system scope.

## Constants

Like with Julia's atomic intrinsics, orderings, scopes and the keywords have to be constants
for the code to be a single instruction. Anything else is a dynamic call, which is slow on
CPUs and invalid in GPU code. To choose one at run time, branch on it and pass a constant in
each branch. Arguments only stay constant through functions that Julia inlines, so mark
functions that forward them `@inline`, or use the primitives for back-ends below.

## Read-modify-write operations

`modify!` is a single `atomicrmw` for these operations, and a loop of `load` and weak `cas!`
otherwise:

| `op` | types | `atomicrmw` |
|:-----|:------|:------------|
| `+`, `-` | integers; `Float16`, `Float32`, `Float64`, and `Core.BFloat16` from LLVM 20 | `add`, `sub`; `fadd`, `fsub` |
| `&`, `\|`, `xor` | integers, `Bool` | `and`, `or`, `xor` |
| `⊼` | integers | `nand` |
| `max`, `min` | integers, `Bool` | `max`, `min`, `umax`, `umin` |
| `max`, `min` | floating-point numbers, from LLVM 21 (Julia 1.14) | `fmaximum`, `fminimum` |
| `UA.fmax`, `UA.fmin` | floating-point numbers (`Core.BFloat16` from LLVM 20) | `fmax`, `fmin` |
| `UA.right` | any | `xchg` |
| `UA.inc_wrap`, `UA.dec_wrap`, `UA.sub_cond`, `UA.sub_sat` | unsigned integers, from LLVM 22 (Julia 1.14) | `uinc_wrap`, `udec_wrap`, `usub_cond`, `usub_sat` |

Every operation is a Julia function with the semantics of the instruction, so the result is the
same with the instruction or the loop. `max` and `min` on floats have Julia's semantics (NaN
propagates, `-0.0 < 0.0`); `UA.fmax` and `UA.fmin` are IEEE `maxNum`/`minNum`, which ignore a
NaN operand. With an instruction, `op(old, x)` in the result of `modify!` is computed in Julia,
which for floating-point numbers can differ from the stored value in NaN payloads or, for
`fmax`/`fmin`, in the sign of zero; `cas!` returns exactly what was stored. `op` has to be
defined for `T`, except for the operations on `Core.BFloat16` in the table above, which
UnsafeAtomics computes itself; the named functions like `add!` only compute the new value when
they need the loop.

## For back-ends

The functions above take keyword arguments and orderings as values, which relies on constant
propagation. Code that passes these arguments through functions that aren't inlined, like the
internals of a GPU back-end, can use the primitives that take everything as a `Val` instead.
They are the same instructions, and part of the public API, like the functions above:

```julia
UA.Internal.llvm_load(ptr, Val(order), Val(scope), Val(volatile), Val(align), Val(md))
UA.Internal.llvm_store!(ptr, x, Val(order), Val(scope), Val(volatile), Val(align), Val(md))
UA.Internal.llvm_rmw!(ptr, Val(op), x, Val(order), Val(scope), Val(volatile), Val(align), Val(md))
UA.Internal.llvm_modify!(ptr, op, x, Val(order), Val(scope), Val(volatile), Val(align), Val(md))
UA.Internal.llvm_cmpxchg!(ptr, cmp, new, Val(success), Val(failure), Val(scope),
                          Val(weak), Val(volatile), Val(align), Val(md))
UA.Internal.llvm_fence(Val(order), Val(scope), Val(md))
```

Orderings are `Symbol`s (LLVM's or Julia's names), the scope is the syncscope's name (`:system`
for the default one), `op` for `llvm_rmw!` an `atomicrmw` operation such as `:uinc_wrap`, and
`md` metadata to attach, `()` or e.g. `((:mmra, ((Symbol("metal-synchronize-as"), :threadgroup),)),)`
for LLVM's memory model relaxation annotations. `llvm_rmw!` emits the operation as it is, so the
target has to support it: e.g., LLVM's AArch64 back-end can't compile `uinc_wrap` before LLVM 22.

## Upgrading from 0.3

- `max` and `min` on floats have Julia's semantics, which only have an instruction from LLVM
  21 (Julia 1.14), and are a compare-and-swap loop before; on `LLVMPtr`, 0.3 used `atomicrmw
  fmax`/`fmin`. Use `UA.fmax`/`UA.fmin` (`fmax!`/`fmin!`) for those instructions' IEEE
  `maxNum`/`minNum`.
- The default scope is called `UA.system`; `UA.none` is an alias. Scopes no longer print as
  LLVM IR.
- `fence(seq_cst)` is a plain `fence` instruction, like `Threads.atomic_fence()`. Before LLVM
  20 (Julia 1.12), it's `mfence` on x86_64, instead of a locked `or`.
- The `LLVMPtr` methods are part of the package instead of an extension: UnsafeAtomics depends
  on LLVM.jl 10, which it uses to generate the instructions.
- The functions are only defined for `Ptr` and `Core.LLVMPtr`, and invalid arguments throw a
  `ConcurrencyViolationError` or an `ArgumentError`.
