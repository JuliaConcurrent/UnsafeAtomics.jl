@inline UnsafeAtomics.load(x) = UnsafeAtomics.load(x, seq_cst)
@inline UnsafeAtomics.store!(x, v) = UnsafeAtomics.store!(x, v, seq_cst)
@inline UnsafeAtomics.cas!(x, cmp, new) = UnsafeAtomics.cas!(x, cmp, new, seq_cst, seq_cst)
@inline UnsafeAtomics.modify!(ptr, op, x) = UnsafeAtomics.modify!(ptr, op, x, seq_cst)

#! format: off
# https://github.com/JuliaLang/julia/blob/v1.6.3/base/atomics.jl#L23-L30
if Sys.ARCH == :i686 || startswith(string(Sys.ARCH), "arm") ||
   Sys.ARCH === :powerpc64le || Sys.ARCH === :ppc64le
    const inttypes = (Int8, Int16, Int32, Int64,
                      UInt8, UInt16, UInt32, UInt64)
else
    const inttypes = (Int8, Int16, Int32, Int64, Int128,
                      UInt8, UInt16, UInt32, UInt64, UInt128)
end
const floattypes = (Float16, Float32, Float64)

# https://github.com/JuliaLang/julia/blob/v1.6.3/base/atomics.jl#L331-L341
const llvmtypes = IdDict{Any,String}(
    Bool => "i8",  # julia represents bools with 8-bits for now. # TODO: is this okay?
    Int8 => "i8", UInt8 => "i8",
    Int16 => "i16", UInt16 => "i16",
    Int32 => "i32", UInt32 => "i32",
    Int64 => "i64", UInt64 => "i64",
    Int128 => "i128", UInt128 => "i128",
    Float16 => "half",
    Float32 => "float",
    Float64 => "double",
)
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
]

for (op, rmwop) in OP_RMW_TABLE
    fn = Symbol(rmwop, "!")
    @eval @inline UnsafeAtomics.$fn(x, v) = UnsafeAtomics.$fn(x, v, seq_cst)
    @eval @inline UnsafeAtomics.$fn(ptr, x, ord) =
        first(UnsafeAtomics.modify!(ptr, $op, x, ord))
end

# Based on: https://github.com/JuliaLang/julia/blob/v1.6.3/base/atomics.jl
for typ in (inttypes..., floattypes...)
    lt = llvmtypes[typ]
    rt = "$lt, $lt*"

    for ord in orderings
        ord in (release, acq_rel) && continue

        @eval function UnsafeAtomics.load(x::Ptr{$typ}, ::$(typeof(ord)))
            return llvmcall(
                $("""
                %ptr = inttoptr i$WORD_SIZE %0 to $lt*
                %rv = load atomic $rt %ptr $ord, align $(sizeof(typ))
                ret $lt %rv
                """),
                $typ,
                Tuple{Ptr{$typ}},
                x,
            )
        end
    end

    for ord in orderings
        ord in (acquire, acq_rel) && continue

        @eval function UnsafeAtomics.store!(x::Ptr{$typ}, v::$typ, ::$(typeof(ord)))
            return llvmcall(
                $("""
                %ptr = inttoptr i$WORD_SIZE %0 to $lt*
                store atomic $lt %1, $lt* %ptr $ord, align $(sizeof(typ))
                ret void
                """),
                Cvoid,
                Tuple{Ptr{$typ},$typ},
                x,
                v,
            )
        end
    end

    for success_ordering in (monotonic, acquire, release, acq_rel, seq_cst),
        failure_ordering in (monotonic, acquire, seq_cst)

        typ <: AbstractFloat && break

        @eval function UnsafeAtomics.cas!(
            x::Ptr{$typ},
            cmp::$typ,
            new::$typ,
            ::$(typeof(success_ordering)),
            ::$(typeof(failure_ordering)),
        )
            success = Ref{Int8}()
            GC.@preserve success begin
                old = llvmcall(
                    $(
                        """
                        %ptr = inttoptr i$WORD_SIZE %0 to $lt*
                        %rs = cmpxchg $lt* %ptr, $lt %1, $lt %2 $success_ordering $failure_ordering
                        %rv = extractvalue { $lt, i1 } %rs, 0
                        %s1 = extractvalue { $lt, i1 } %rs, 1
                        %s8 = zext i1 %s1 to i8
                        %sptr = inttoptr i$WORD_SIZE %3 to i8*
                        store i8 %s8, i8* %sptr
                        ret $lt %rv
                        """
                    ),
                    $typ,
                    Tuple{Ptr{$typ},$typ,$typ,Ptr{Int8}},
                    x,
                    cmp,
                    new,
                    Ptr{Int8}(pointer_from_objref(success)),
                )
            end
            return (old = old, success = !iszero(success[]))
        end
    end

    for (op, rmwop) in OP_RMW_TABLE
        rmw = string(rmwop)
        fn = Symbol(rmw, "!")
        if (rmw == "max" || rmw == "min") && typ <: Unsigned
            # LLVM distinguishes signedness in the operation, not the integer type.
            rmw = "u" * rmw
        end
        if typ <: AbstractFloat
            if rmw == "add"
                rmw = "fadd"
            elseif rmw == "sub"
                rmw = "fsub"
            else
                continue
            end
        end
        for ord in orderings
            @eval function UnsafeAtomics.modify!(
                x::Ptr{$typ},
                ::typeof($op),
                v::$typ,
                ::$(typeof(ord)),
            )
                old = llvmcall(
                    $("""
                    %ptr = inttoptr i$WORD_SIZE %0 to $lt*
                    %rv = atomicrmw $rmw $lt* %ptr, $lt %1 $ord
                    ret $lt %rv
                    """),
                    $typ,
                    Tuple{Ptr{$typ},$typ},
                    x,
                    v,
                )
                return old => $op(old, v)
            end
        end
    end

end

function UnsafeAtomics.cas!(
    x::Ptr{T},
    cmp::T,
    new::T,
    success_ordering,
    failure_ordering,
) where {T}
    if sizeof(T) == 1
        (old, success) = UnsafeAtomics.cas!(
            Ptr{UInt8}(x),
            bitcast(UInt8, cmp),
            bitcast(UInt8, new),
            success_ordering,
            failure_ordering,
        )
        return (old = bitcast(T, old), success = success)
    elseif sizeof(T) == 2
        (old, success) = UnsafeAtomics.cas!(
            Ptr{UInt16}(x),
            bitcast(UInt16, cmp),
            bitcast(UInt16, new),
            success_ordering,
            failure_ordering,
        )
        return (old = bitcast(T, old), success = success)
    elseif sizeof(T) == 4
        (old, success) = UnsafeAtomics.cas!(
            Ptr{UInt32}(x),
            bitcast(UInt32, cmp),
            bitcast(UInt32, new),
            success_ordering,
            failure_ordering,
        )
        return (old = bitcast(T, old), success = success)
    elseif sizeof(T) == 8
        (old, success) = UnsafeAtomics.cas!(
            Ptr{UInt64}(x),
            bitcast(UInt64, cmp),
            bitcast(UInt64, new),
            success_ordering,
            failure_ordering,
        )
        return (old = bitcast(T, old), success = success)
    elseif sizeof(T) == 16
        (old, success) = UnsafeAtomics.cas!(
            Ptr{UInt128}(x),
            bitcast(UInt128, cmp),
            bitcast(UInt128, new),
            success_ordering,
            failure_ordering,
        )
        return (old = bitcast(T, old), success = success)
    else
        error(LazyString("unsupported size: ", sizeof(T)))
    end
end
