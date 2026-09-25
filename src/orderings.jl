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

normalize_order(o) = o === :acquire_release ? :acq_rel : o === :sequentially_consistent ? :seq_cst : o

# The LLVM name of an ordering, as a `Val` for the generator, which rejects invalid ones. Like
# for Julia's atomic intrinsics, the ordering has to be a constant: otherwise, constructing the
# `Val` is a dynamic call.
@inline ordering_val(::LLVMOrdering{name}) where {name} = Val(name)
@inline ordering_val(order::Symbol) = Val(normalize_order(order))
ordering_val(@nospecialize(order)) = throw_invalid_ordering()

@noinline throw_invalid_ordering() =
    throw(Base.ConcurrencyViolationError("invalid atomic ordering"))
