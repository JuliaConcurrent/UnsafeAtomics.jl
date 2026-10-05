module UnsafeAtomicsGPUCompilerExt

using GPUCompiler: GPUCompiler
using UnsafeAtomics: Internal

# GPUCompiler legalizes atomics for the target it compiles for, so emit the instruction that
# implements an operation, even where the host's back-end couldn't compile it.
Base.Experimental.@overlay GPUCompiler.SHARED_METHOD_TABLE Internal.is_native() = false

end
