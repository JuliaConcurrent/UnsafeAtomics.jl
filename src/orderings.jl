struct LLVMOrdering{name} <: Ordering end

const unordered = LLVMOrdering{:unordered}()
const monotonic = LLVMOrdering{:monotonic}()
const acquire = LLVMOrdering{:acquire}()
const release = LLVMOrdering{:release}()
const acq_rel = LLVMOrdering{:acq_rel}()
const seq_cst = LLVMOrdering{:seq_cst}()

const orderings = (unordered, monotonic, acquire, release, acq_rel, seq_cst)

const ConcreteOrdering = Union{map(typeof, orderings)...}

llvm_ordering(::LLVMOrdering{name}) where {name} = name

Base.string(o::LLVMOrdering) = String(llvm_ordering(o))
Base.print(io::IO, o::LLVMOrdering) = print(io, string(o))

Base.show(io::IO, o::ConcreteOrdering) = print(io, UnsafeAtomics, '.', llvm_ordering(o))

base_ordering(::LLVMOrdering{name}) where {name} = name
base_ordering(::LLVMOrdering{:seq_cst}) = :sequentially_consistent
base_ordering(::LLVMOrdering{:acq_rel}) = :acquire_release

# The failure ordering of a cmpxchg can't release. Derive it from the success ordering
# like C++ does.
@inline failure_order(order) =
    (order === release || order === :release) ? monotonic :
    (order === acq_rel || order === :acq_rel || order === :acquire_release) ? acquire : order

# Call `f(Val(name), args...)` with the LLVM name of the ordering. Orderings are selected by
# branching, not by dispatch, so that an ordering that isn't a constant, e.g. a `Union` of
# several of them or a `Symbol`, results in a branch per ordering rather than in a dynamic
# call (which a GPU can't do). A constant ordering leaves a single branch. Comparing with `===`
# only compares pointers, even when Julia widens the `Union` to the abstract type, whereas
# `isa` would load the type from the object, i.e. from host memory on a GPU. (The type
# parameters make Julia specialize on `f` and `args`, which it doesn't when only passing them.)
@inline with_ordering(f::F, order, args::Vararg{Any,N}) where {F,N} =
    (order === monotonic || order === :monotonic) ? f(Val(:monotonic), args...) :
    (order === acquire || order === :acquire) ? f(Val(:acquire), args...) :
    (order === release || order === :release) ? f(Val(:release), args...) :
    (order === acq_rel || order === :acq_rel || order === :acquire_release) ?
        f(Val(:acq_rel), args...) :
    (order === seq_cst || order === :seq_cst || order === :sequentially_consistent) ?
        f(Val(:seq_cst), args...) :
    (order === unordered || order === :unordered) ? f(Val(:unordered), args...) :
    throw_invalid_ordering()

# Without arguments: passing an `order` that is a `Union` would make this a dynamic call.
@noinline throw_invalid_ordering() =
    throw(Base.ConcurrencyViolationError("invalid atomic ordering"))
