module Helpers

using UnsafeAtomics.Internal: ATOMIC_SIZES
using Core: LLVMPtr
using InteractiveUtils: code_llvm

export inttypes, floattypes, POINTER_KINDS, pointer_to, llvm_ir, AbstractBits, asbits

if 16 in ATOMIC_SIZES
    const inttypes = (Int8, Int16, Int32, Int64, Int128,
                      UInt8, UInt16, UInt32, UInt64, UInt128)
else
    const inttypes = (Int8, Int16, Int32, Int64,
                      UInt8, UInt16, UInt32, UInt64)
end
const floattypes = (Float16, Float32, Float64)

# The pointer kinds to run on (the CPU can only access address space 0).
const POINTER_KINDS = (Ptr, LLVMPtr)
pointer_to(::Type{Ptr}, xs, i = 1) = pointer(xs, i)
pointer_to(::Type{LLVMPtr}, xs, i = 1) = reinterpret(LLVMPtr{eltype(xs),0}, pointer(xs, i))

# The generated code, without the counters that code coverage adds (`atomicrmw add` on a
# constant address).
llvm_ir(f, types; kwargs...) =
    join(filter(!contains("inttoptr ("),
                split(sprint(io -> code_llvm(io, f, types; debuginfo = :none, kwargs...)), '\n')), '\n')

# primitive types without arithmetic
abstract type AbstractBits end

function asuint end
function asbits end

for T in [UInt8, UInt16, UInt32, UInt64, UInt128]
    C = :(Base.$(nameof(T)))
    nbits = sizeof(T) * 8
    B = Symbol("Bits$(nbits)")
    @eval begin
        export $B
        primitive type $B <: AbstractBits $nbits end
        asbits(::Type{$T}) = $B
        asuint(::Type{$B}) = $T
        $B(x::$T) = reinterpret($B, x)
        $C(x::$B) = reinterpret($T, x)
    end
end

Base.rand(::Type{B}) where {B<:AbstractBits} = B(rand(asuint(B)))

end  # module
