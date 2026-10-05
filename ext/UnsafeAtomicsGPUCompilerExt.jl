module UnsafeAtomicsGPUCompilerExt

using GPUCompiler: GPUCompiler
using UnsafeAtomics: Internal

# GPUCompiler legalizes atomics for the target it compiles for, so emit the instruction that
# implements an operation, and the scope that was asked for, even where the host's back-end
# couldn't compile them.
Base.Experimental.@overlay GPUCompiler.SHARED_METHOD_TABLE Internal.is_native() = false

end
