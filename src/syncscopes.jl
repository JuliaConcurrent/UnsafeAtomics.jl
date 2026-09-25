struct LLVMSyncScope{name} <: SyncScope end

const none = LLVMSyncScope{:none}()
const singlethread = LLVMSyncScope{:singlethread}()
const subgroup = LLVMSyncScope{:subgroup}()
const workgroup = LLVMSyncScope{:workgroup}()
const device = LLVMSyncScope{:device}()
const system = none

const syncscopes = (none, singlethread)
const ConcreteSyncScopes = Union{map(typeof, (none, singlethread, subgroup, workgroup, device))...}

"""
    UnsafeAtomics.SyncScope(name::Symbol)

The LLVM synchronization scope called `name`. The canonical scopes are available as
`UnsafeAtomics.singlethread`, `subgroup`, `workgroup`, `device` and `system`, which is the
default and the same object as `UnsafeAtomics.none`. Other names, e.g. `SyncScope(:agent)`,
are passed to LLVM verbatim, for scopes that are specific to a target.
"""
UnsafeAtomics.SyncScope(name::Symbol) = name === :system ? system : LLVMSyncScope{name}()

llvm_syncscope(::LLVMSyncScope{name}) where {name} = name

# The name ends up in an LLVM string literal, so escape the characters that would end it.
llvm_string(name) = replace(String(name), '\\' => "\\5C", '"' => "\\22")

Base.string(s::LLVMSyncScope) = string("syncscope(\"", llvm_string(llvm_syncscope(s)), "\")")
Base.string(s::typeof(none)) = ""

Base.print(io::IO, s::LLVMSyncScope) = print(io, string(s))

Base.show(io::IO, o::ConcreteSyncScopes) = print(io, UnsafeAtomics, '.', llvm_syncscope(o))
Base.show(io::IO, o::LLVMSyncScope) =
    print(io, UnsafeAtomics, ".SyncScope(", repr(llvm_syncscope(o)), ')')
