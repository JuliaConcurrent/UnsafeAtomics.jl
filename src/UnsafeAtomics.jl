baremodule UnsafeAtomics

abstract type Ordering end

function load end
function store! end
function cas! end
function modify! end

function add! end
function sub! end
function xchg! end
function and! end
function nand! end
function or! end
function xor! end
function max! end
function min! end

right(_, x) = x

module Internal

using Base.Sys: WORD_SIZE
using Base: llvmcall

using ..UnsafeAtomics: UnsafeAtomics, Ordering, right

include("utils.jl")
include("orderings.jl")
include("core.jl")

end  # module Internal

const unordered = Internal.unordered
const monotonic = Internal.monotonic
const acquire = Internal.acquire
const release = Internal.release
const acq_rel = Internal.acq_rel
const seq_cst = Internal.seq_cst

# Julia names
const acquire_release = acq_rel
const sequentially_consistent = seq_cst

end  # baremodule UnsafeAtomics
