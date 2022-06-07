module TestCore

using UnsafeAtomics: UnsafeAtomics, acquire, release, acq_rel
using UnsafeAtomics.Internal: OP_RMW_TABLE, OP_RMW_FP_TABLE, inttypes, floattypes
using Test

function test_default_ordering()
    @testset for T in inttypes
        test_default_ordering(T)
    end
    @testset for T in floattypes
        test_default_ordering(T)
    end
end

function test_default_ordering(T::Type)
    xs = T[rand(T), rand(T)]
    x1 = rand(T)
    x2 = rand(T)
    @debug "xs=$(repr(xs)) x1=$(repr(x1)) x2=$(repr(x2))"

    ptr = pointer(xs, 1)
    GC.@preserve xs begin
        @test UnsafeAtomics.load(ptr) === xs[1]
        UnsafeAtomics.store!(ptr, x1)
        @test xs[1] === x1
        desired = (old = x1, success = true)
        @test UnsafeAtomics.cas!(ptr, x1, x2) === (old = x1, success = true)
        @test xs[1] === x2
        table = if T <: Integer
            OP_RMW_TABLE
        elseif T <: AbstractFloat
            OP_RMW_FP_TABLE
        end
        @testset for (op, name) in table
            xs[1] = x1
            @test UnsafeAtomics.modify!(ptr, op, x2) === (x1 => op(x1, x2))
            @test xs[1] === op(x1, x2)

            rmw = getfield(UnsafeAtomics, Symbol(name, :!))
            xs[1] = x1
            @test rmw(ptr, x2) === x1
            @test xs[1] === op(x1, x2)
        end
    end
end

function test_explicit_ordering()
    test_explicit_ordering(UInt)
    test_explicit_ordering(Float64)
end

function test_explicit_ordering(T::Type)
    xs = T[rand(T), rand(T)]
    x1 = rand(T)
    x2 = rand(T)
    @debug "xs=$(repr(xs)) x1=$(repr(x1)) x2=$(repr(x2))"

    ptr = pointer(xs, 1)
    GC.@preserve xs begin

        @test UnsafeAtomics.load(ptr, acquire) === xs[1]
        UnsafeAtomics.store!(ptr, x1, release)
        @test xs[1] === x1
        desired = (old = x1, success = true)
        @test UnsafeAtomics.cas!(ptr, x1, x2, acq_rel, acquire) === desired
        @test xs[1] === x2
        table = if T <: Integer
            OP_RMW_TABLE
        elseif T <: AbstractFloat
            OP_RMW_FP_TABLE
        end
        @testset for (op, name) in table
            xs[1] = x1
            @test UnsafeAtomics.modify!(ptr, op, x2, acq_rel) === (x1 => op(x1, x2))
            @test xs[1] === op(x1, x2)

            rmw = getfield(UnsafeAtomics, Symbol(name, :!))
            xs[1] = x1
            @test rmw(ptr, x2, acquire) === x1
            @test xs[1] === op(x1, x2)
        end
    end
end

end  # module
