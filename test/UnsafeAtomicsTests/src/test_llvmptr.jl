module TestLLVMPtr

import InteractiveUtils

using UnsafeAtomics: UnsafeAtomics, acquire, release, acq_rel, seq_cst
using UnsafeAtomics.Internal: OP_RMW_TABLE, inttypes
using Test
using Base: ConcurrencyViolationError

# the operations that apply to integers
const INT_RMW_TABLE = [(op, name) for (op, name) in OP_RMW_TABLE
                       if !(op in (UnsafeAtomics.fmax, UnsafeAtomics.fmin))]

llvmptr(xs::Array, i) = reinterpret(Core.LLVMPtr{eltype(xs),0}, pointer(xs, i))

function check_default_ordering(T::Type)
    xs = T[rand(T), rand(T)]
    x1 = rand(T)
    x2 = rand(T)
    check_default_ordering(xs, x1, x2)
end

function check_default_ordering(xs::AbstractArray{T}, x1::T, x2::T) where T
    @debug "xs=$(repr(xs)) x1=$(repr(x1)) x2=$(repr(x2))"

    ptr = llvmptr(xs, 1)
    GC.@preserve xs begin
        @test UnsafeAtomics.load(ptr) === xs[1]
        UnsafeAtomics.store!(ptr, x1)
        @test xs[1] === x1
        sizeof(T) == 0 && return # CAS hangs on zero sized data...
        desired = (old = x1, success = true)
        @test UnsafeAtomics.cas!(ptr, x1, x2) === (old = x1, success = true)
        @test xs[1] === x2
        @testset for (op, name) in INT_RMW_TABLE
            xs[1] = x1
            @test UnsafeAtomics.modify!(ptr, op, x2) === (x1 => op(x1, x2))
            @test xs[1] === op(x1, x2)

            rmw = getfield(UnsafeAtomics, Symbol(name, :!))
            xs[1] = x1
            @test rmw(ptr, x2) === x1
            @test xs[1] === op(x1, x2)

            # Check dispatch to LLVM atomic OP instead of CAS loop.
            if (op == +) || (op == -)
                IR = sprint(io->InteractiveUtils.code_llvm(io,
                    UnsafeAtomics.modify!,
                    typeof.((ptr, +, T(1)))))
                @test occursin("atomicrmw", IR)
            end
        end
    end
end

function check_explicit_ordering(T::Type = UInt)
    xs = T[rand(T), rand(T)]
    x1 = rand(T)
    x2 = rand(T)
    check_explicit_ordering(xs, x1, x2)
end

function check_explicit_ordering(xs::AbstractArray{T}, x1::T, x2::T) where T
    @debug "xs=$(repr(xs)) x1=$(repr(x1)) x2=$(repr(x2))"

    ptr = llvmptr(xs, 1)
    GC.@preserve xs begin

        @test UnsafeAtomics.load(ptr, acquire) === xs[1]
        UnsafeAtomics.store!(ptr, x1, release)
        @test xs[1] === x1
        sizeof(T) == 0 && return # CAS hangs on zero sized data...
        desired = (old = x1, success = true)
        @test UnsafeAtomics.cas!(ptr, x1, x2, acq_rel, acquire) === desired
        @test xs[1] === x2
        @testset for (op, name) in INT_RMW_TABLE
            xs[1] = x1
            @test UnsafeAtomics.modify!(ptr, op, x2, acq_rel) === (x1 => op(x1, x2))
            @test xs[1] === op(x1, x2)

            rmw = getfield(UnsafeAtomics, Symbol(name, :!))
            xs[1] = x1
            @test rmw(ptr, x2, acquire) === x1
            @test xs[1] === op(x1, x2)

            # Test syncscopes.
            if (op == +) || (op == -)
                xs[1] = x1
                @test UnsafeAtomics.modify!(ptr, op, x2, seq_cst, UnsafeAtomics.none) ===
                      (x1 => op(x1, x2))
                @test xs[1] === op(x1, x2)

                xs[1] = x1
                @test UnsafeAtomics.modify!(ptr, op, x2, seq_cst, UnsafeAtomics.singlethread) ===
                      (x1 => op(x1, x2))
                @test xs[1] === op(x1, x2)
            end

            # Check dispatch to LLVM atomic OP instead of CAS loop.
            if (op == +) || (op == -)
                IR = sprint(io->InteractiveUtils.code_llvm(io,
                    UnsafeAtomics.modify!,
                    typeof.((ptr, +, T(1), seq_cst, UnsafeAtomics.singlethread))))
                @test occursin("atomicrmw", IR)
            end
        end
    end
end


# The LLVMPtr methods used to live in a package extension that needed LLVM.jl.
function test_without_llvm()
    @test !haskey(Base.loaded_modules, Base.PkgId(Base.UUID("929cbde3-209d-540e-8aea-75f648917ca0"), "LLVM"))
    @test Base.return_types(UnsafeAtomics.add!, (Core.LLVMPtr{Int32,1}, Int32)) == [Int32]
end

function test_default_ordering()
    @testset for T in inttypes
        check_default_ordering(T)
    end
end

function test_explicit_ordering()
    @testset for T in inttypes
        check_explicit_ordering(T)
    end
end

function test_cas_single_ordering()
    @testset for ord in [UnsafeAtomics.monotonic, acquire, release, acq_rel, seq_cst]
        xs = Int32[1, 2]
        ptr = llvmptr(xs, 1)
        GC.@preserve xs begin
            @test UnsafeAtomics.cas!(ptr, Int32(1), Int32(3), ord) ===
                  (old = Int32(1), success = true)
            @test UnsafeAtomics.cas!(ptr, Int32(1), Int32(4), ord) ===
                  (old = Int32(3), success = false)
            @test xs[1] === Int32(3)
        end
    end
end

function test_unordered_rmw()
    xs = Int32[1, 2]
    ptr = llvmptr(xs, 1)
    GC.@preserve xs begin
        @test_throws ConcurrencyViolationError UnsafeAtomics.add!(
            ptr, Int32(1), UnsafeAtomics.unordered)
        @test_throws ConcurrencyViolationError UnsafeAtomics.modify!(
            ptr, *, Int32(1), UnsafeAtomics.unordered)
        @test xs == Int32[1, 2]
    end
end

function test_syncscopes()
    # the LLVMPtr path supports any scope; check that it ends up in the IR
    scopes = [UnsafeAtomics.singlethread, UnsafeAtomics.subgroup, UnsafeAtomics.workgroup,
              UnsafeAtomics.device, UnsafeAtomics.system, UnsafeAtomics.SyncScope(:agent)]
    @testset for scope in scopes, AS in [0, 1, 3]
        P = Core.LLVMPtr{Int32,AS}
        S = typeof(scope)
        name = UnsafeAtomics.Internal.llvm_syncscope(scope)
        expected = scope === UnsafeAtomics.system ? r"^((?!syncscope).)*$" :
                   Regex("syncscope\\(\"$name\"\\)")
        for (f, types, instruction) in [
                ((p, s) -> UnsafeAtomics.load(p, acquire, s), (P, S), "load atomic"),
                ((p, x, s) -> UnsafeAtomics.store!(p, x, release, s), (P, Int32, S), "store atomic"),
                ((p, c, n, s) -> UnsafeAtomics.cas!(p, c, n, acq_rel, acquire, s), (P, Int32, Int32, S), "cmpxchg"),
                ((p, x, s) -> UnsafeAtomics.add!(p, x, acq_rel, s), (P, Int32, S), "atomicrmw add")]
            ir = sprint(io -> InteractiveUtils.code_llvm(io, f, types; debuginfo = :none))
            # (without coverage counters, which are `atomicrmw add` on a constant address)
            line = only(filter(l -> contains(l, instruction) && !contains(l, "inttoptr ("),
                               split(ir, '\n')))
            @test occursin(expected, line)
        end
    end
end

# from memory: before LLVM 17, the AArch64 back-end can't materialize a bfloat constant
const BFLOAT16_BITS = [0x3f80, 0x4000]  # 1.0, 2.0

function test_bfloat16()
    isdefined(Core, :BFloat16) || return
    one, two = reinterpret(Core.BFloat16, BFLOAT16_BITS)
    bits(x) = reinterpret(UInt16, x)
    xs = [one]
    ptr = llvmptr(xs, 1)
    GC.@preserve xs begin
        @test bits(UnsafeAtomics.xchg!(ptr, two)) === 0x3f80
        if Base.libllvm_version >= v"20"
            # Core.BFloat16 has no arithmetic without BFloat16s.jl, but `add!` doesn't need it
            @test bits(UnsafeAtomics.add!(ptr, one)) === 0x4000
            @test bits(xs[1]) === 0x4040  # 3.0
        else
            @test UnsafeAtomics.Internal.native_rmw(+, Core.BFloat16) === nothing
        end
    end
end

function test_zero_size()
    @test sizeof(Nothing) == 0
    check_default_ordering([nothing, nothing], nothing, nothing)
    check_explicit_ordering([nothing, nothing], nothing, nothing)
end

end  # module
