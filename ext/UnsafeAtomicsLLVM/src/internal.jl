# TODO: move this to UnsafeAtomics
julia_ordering_name(::UnsafeAtomics.Internal.LLVMOrdering{name}) where {name} = name
julia_ordering_name(::typeof(UnsafeAtomics.acquire_release)) = :acquire_release
julia_ordering_name(::typeof(UnsafeAtomics.sequentially_consistent)) =
    :sequentially_consistent

@inline UnsafeAtomics.load(ptr::LLVMPtr, order::Ordering) =
    LLVM.Interop.atomic_pointerref(ptr, Val{julia_ordering_name(order)}())

@inline function UnsafeAtomics.store!(ptr::LLVMPtr, x, order::Ordering)
    LLVM.Interop.atomic_pointerset(ptr, x, Val{julia_ordering_name(order)}())
    return
end

mapop(op::OP) where {OP} = op
mapop(::typeof(UnsafeAtomics.right)) = LLVM.Interop.right

@inline UnsafeAtomics.modify!(ptr::LLVMPtr, op::OP, x, order::Ordering) where {OP} =
    LLVM.Interop.atomic_pointermodify(ptr, mapop(op), x, Val{julia_ordering_name(order)}())

@inline UnsafeAtomics.cas!(
    ptr::LLVMPtr,
    expected,
    desired,
    success_order::Ordering,
    failure_order::Ordering,
) = LLVM.Interop.atomic_pointerreplace(
    ptr,
    expected,
    desired,
    Val{julia_ordering_name(success_order)}(),
    Val{julia_ordering_name(failure_order)}(),
)
