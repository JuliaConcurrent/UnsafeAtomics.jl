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
    @eval @inline UnsafeAtomics.$fn(x, v, ord) = UnsafeAtomics.$fn(x, v, ord, none) 
    @eval @inline UnsafeAtomics.$fn(ptr, x, ord, scope) =
        first(UnsafeAtomics.modify!(ptr, $op, x, ord, scope))
end

const ATOMIC_INTRINSICS = isdefined(Core.Intrinsics, :atomic_pointerref)

if VERSION >= v"1.12.0-DEV.161" && Int == Int64
const MAX_ATOMIC_SIZE = 16
const MAX_POINTERATOMIC_SIZE = 16
else
const MAX_ATOMIC_SIZE = 8
const MAX_POINTERATOMIC_SIZE = 8
end


if VERSION < v"1.12.0"
    ptr(typ) = typ*"*"
    inttoptr(typ, arg) = "inttoptr i$WORD_SIZE $arg to $(ptr(typ))"
else
    ptr(typ) = "ptr"
    inttoptr(_, arg) = "bitcast ptr $arg to ptr"
end

# Based on: https://github.com/JuliaLang/julia/blob/v1.6.3/base/atomics.jl
for typ in (inttypes..., floattypes...)
    lt = llvmtypes[typ]
    rt = "$lt, $(ptr(lt))"

    for ord in orderings
        ord in (release, acq_rel) && continue

        for sync in syncscopes 
            if ATOMIC_INTRINSICS && sizeof(typ) <= MAX_POINTERATOMIC_SIZE && sync == none
                @eval function UnsafeAtomics.load(x::Ptr{$typ}, ::$(typeof(ord)), ::$(typeof(sync)))
                    return Core.Intrinsics.atomic_pointerref(x, base_ordering($ord))
                end
            else
                @eval function UnsafeAtomics.load(x::Ptr{$typ}, ::$(typeof(ord)), ::$(typeof(sync)))
                    return llvmcall(
                        $("""
                        %ptr = $(inttoptr(lt, "%0"))
                        %rv = load atomic $rt %ptr $sync $ord, align $(sizeof(typ))
                        ret $lt %rv
                        """),
                        $typ,
                        Tuple{Ptr{$typ}},
                        x,
                    )
                end
            end
        end
    end

    for ord in orderings
        ord in (acquire, acq_rel) && continue
        
        for sync in syncscopes 
            if ATOMIC_INTRINSICS && sizeof(typ) <= MAX_POINTERATOMIC_SIZE && sync == none
                @eval function UnsafeAtomics.store!(x::Ptr{$typ}, v::$typ, ::$(typeof(ord)), ::$(typeof(sync)))
                    Core.Intrinsics.atomic_pointerset(x, v, base_ordering($ord))
                    return nothing
                end
            else
                @eval function UnsafeAtomics.store!(x::Ptr{$typ}, v::$typ, ::$(typeof(ord)), ::$(typeof(sync)))
                    return llvmcall(
                        $("""
                        %ptr = $(inttoptr(lt, "%0"))
                        store atomic $lt %1, $(ptr(lt)) %ptr $sync $ord, align $(sizeof(typ))
                        ret void
                        """),
                        Cvoid,
                        Tuple{Ptr{$typ},$typ},
                        x,
                        v,
                    )
                end
            end
        end
    end

    for success_ordering in (monotonic, acquire, release, acq_rel, seq_cst),
        failure_ordering in (monotonic, acquire, seq_cst)

        typ <: AbstractFloat && break

        for sync in syncscopes 
            if ATOMIC_INTRINSICS && sizeof(typ) <= MAX_POINTERATOMIC_SIZE && sync == none
                @eval function UnsafeAtomics.cas!(
                    x::Ptr{$typ},
                    cmp::$typ,
                    new::$typ,
                    ::$(typeof(success_ordering)),
                    ::$(typeof(failure_ordering)),
                    ::$(typeof(sync)),
                )
                    return Core.Intrinsics.atomic_pointerreplace(
                        x,
                        cmp,
                        new,
                        base_ordering($success_ordering),
                        base_ordering($failure_ordering)
                    )
                end
            else
                @eval function UnsafeAtomics.cas!(
                    x::Ptr{$typ},
                    cmp::$typ,
                    new::$typ,
                    ::$(typeof(success_ordering)),
                    ::$(typeof(failure_ordering)),
                    ::$(typeof(sync)),
                )
                    success = Ref{Int8}()
                    GC.@preserve success begin
                        old = llvmcall(
                            $(
                                """
                                %ptr = $(inttoptr(lt, "%0"))
                                %rs = cmpxchg $(ptr(lt)) %ptr, $lt %1, $lt %2 $sync $success_ordering $failure_ordering
                                %rv = extractvalue { $lt, i1 } %rs, 0
                                %s1 = extractvalue { $lt, i1 } %rs, 1
                                %s8 = zext i1 %s1 to i8
                                %sptr = $(inttoptr("i8", "%3"))
                                store i8 %s8, $(ptr("i8")) %sptr
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
            for sync in syncscopes
                # Enable this code iff https://github.com/JuliaLang/julia/pull/45122 get's merged
                if false && ATOMIC_INTRINSICS && sizeof(typ) <= MAX_POINTERATOMIC_SIZE && sync == none
                    @eval function UnsafeAtomics.modify!(
                            x::Ptr{$typ},
                            op::typeof($op),
                            v::$typ,
                            ::$(typeof(ord)),
                            ::$(typeof(sync)),
                        )
                            return Core.Intrinsics.atomic_pointermodify(x, op, v, base_ordering($ord))
                    end
                else
                    @eval function UnsafeAtomics.modify!(
                        x::Ptr{$typ},
                        ::typeof($op),
                        v::$typ,
                        ::$(typeof(ord)),
                        ::$(typeof(sync)),
                    )
                        old = llvmcall(
                            $("""
                            %ptr = $(inttoptr(lt, "%0"))
                            %rv = atomicrmw $rmw $(ptr(lt)) %ptr, $lt %1 $sync $ord
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
    end
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

for sync in syncscopes
    if sync == none
        if !FENCE_INTRINSIC_ELIDABLE
            # Core.Intrinsics.atomic_fence was introduced in 1.10
            if VERSION < v"1.14.0-DEV.1371"
                @eval function UnsafeAtomics.fence(ord::Ordering, ::$(typeof(sync)))
                    Core.Intrinsics.atomic_fence(base_ordering(ord))
                    return nothing
                end
            else
                @eval function UnsafeAtomics.fence(ord::Ordering, ::$(typeof(sync)))
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
                    @eval UnsafeAtomics.fence(::$(typeof(ord)), ::$(typeof(sync))) = nothing
                elseif ord === unordered
                    # defined below
                elseif ord === seq_cst && Sys.ARCH == :x86_64
                    # defined by the x86_64 special case below
                else
                    @eval function UnsafeAtomics.fence(::$(typeof(ord)), ::$(typeof(sync)))
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
        # A fence can't be unordered. Throw the intrinsic's error ourselves: Julia's
        # inference thinks the intrinsic throws another type of exception, which Julia 1.11
        # miscompiles when the error is caught. A call that always throws can't be elided.
        @eval UnsafeAtomics.fence(::typeof(unordered), ::$(typeof(sync))) =
            throw_invalid_ordering()
        if Sys.ARCH == :x86_64
            # FIXME: Disable this once on LLVM 19
            # This is unfortunatly required for good-performance on AMD
            # https://github.com/llvm/llvm-project/pull/106555
            @eval function UnsafeAtomics.fence(::typeof(seq_cst), ::$(typeof(sync)))
                Base.llvmcall(
                    (raw"""
                    define void @fence() #0 {
                    entry:
                        tail call void asm sideeffect "lock orq $$0 , (%rsp)", ""(); should this have ~{memory}
                        ret void
                    }
                    attributes #0 = { alwaysinline }
                    """, "fence"), Nothing, Tuple{})
            end
        end
    else
        for ord in orderings
            @eval function UnsafeAtomics.fence(::$(typeof(ord)), ::$(typeof(sync)))
                return llvmcall(
                    $("""
                    fence $sync $ord
                    ret void
                    """),
                    Cvoid,
                    Tuple{},
                )
            end
        end
    end
end

as_native_uint(::Type{T}) where {T} =
    if sizeof(T) == 1
        UInt8
    elseif sizeof(T) == 2
        UInt16
    elseif sizeof(T) == 4
        UInt32
    elseif sizeof(T) == 8
        UInt64
    elseif sizeof(T) == 16
        UInt128
    else
        error(LazyString("unsupported size: ", sizeof(T)))
    end

const LOAD_ORDERINGS = (unordered, monotonic, acquire, seq_cst)
const STORE_ORDERINGS = (unordered, monotonic, release, seq_cst)
const RMW_ORDERINGS = (unordered, monotonic, acquire, release, acq_rel, seq_cst)
const CAS_SUCCESS_ORDERINGS = (monotonic, acquire, release, acq_rel, seq_cst)
const CAS_FAILURE_ORDERINGS = (monotonic, acquire, seq_cst)

# The fallbacks below retry with a same-sized unsigned integer. If `T` already is that
# integer, no specialized method matched: report why, instead of recursing forever.
@noinline function throw_unsupported(::Type{T}, syncscope, valid_ordering::Bool) where {T}
    valid_ordering || throw_invalid_ordering()
    syncscope in syncscopes ||
        throw(ArgumentError(string("unsupported syncscope: ", repr(syncscope))))
    throw(ArgumentError(string("unsupported atomic type: ", T)))
end

function UnsafeAtomics.load(x::Ptr{T}, ordering, syncscope) where {T}
    UI = as_native_uint(T)
    UI === T && throw_unsupported(T, syncscope, ordering in LOAD_ORDERINGS)
    v = UnsafeAtomics.load(Ptr{UI}(x), ordering, syncscope)
    return bitcast(T, v)
end

function UnsafeAtomics.store!(x::Ptr{T}, v::T, ordering, syncscope) where {T}
    UI = as_native_uint(T)
    UI === T && throw_unsupported(T, syncscope, ordering in STORE_ORDERINGS)
    UnsafeAtomics.store!(Ptr{UI}(x), bitcast(UI, v), ordering, syncscope)::Nothing
end

function UnsafeAtomics.modify!(x::Ptr{T}, ::typeof(right), v::T, ordering, syncscope) where {T}
    UI = as_native_uint(T)
    UI === T && throw_unsupported(T, syncscope, ordering in RMW_ORDERINGS)
    old, _ = UnsafeAtomics.modify!(Ptr{UI}(x), right, bitcast(UI, v), ordering, syncscope)
    return bitcast(T, old) => v
end

function UnsafeAtomics.cas!(
    x::Ptr{T},
    cmp::T,
    new::T,
    success_ordering,
    failure_ordering,
    syncscope,
) where {T}
    UI = as_native_uint(T)
    UI === T && throw_unsupported(
        T,
        syncscope,
        success_ordering in CAS_SUCCESS_ORDERINGS && failure_ordering in CAS_FAILURE_ORDERINGS,
    )
    (old, success) = UnsafeAtomics.cas!(
        Ptr{UI}(x),
        bitcast(UI, cmp),
        bitcast(UI, new),
        success_ordering,
        failure_ordering,
        syncscope
    )
    return (old = bitcast(T, old), success = success)
end
