struct LLVMSyncScope{name} <: SyncScope end

const singlethread = LLVMSyncScope{:singlethread}()
const subgroup = LLVMSyncScope{:subgroup}()
const workgroup = LLVMSyncScope{:workgroup}()
const device = LLVMSyncScope{:device}()
const system = LLVMSyncScope{:system}()
const none = system  # its name before 0.4

const ConcreteSyncScopes =
    Union{map(typeof, (singlethread, subgroup, workgroup, device, system))...}

"""
    UnsafeAtomics.SyncScope(name::Symbol)

The LLVM synchronization scope called `name`. The canonical scopes are available as
`UnsafeAtomics.singlethread`, `subgroup`, `workgroup`, `device` and `system`, which is the
default (in LLVM IR, an instruction without a syncscope). Other names, e.g.
`SyncScope(:agent)`, are passed to LLVM verbatim, for scopes that are specific to a target.
"""
UnsafeAtomics.SyncScope(name::Symbol) = LLVMSyncScope{name}()

# The name of the scope, which the generator also uses, with `:system` for the default one.
llvm_syncscope(::LLVMSyncScope{name}) where {name} = name

# The name ends up in an LLVM string literal, so escape the characters that would end it.
llvm_string(name) = replace(String(name), '\\' => "\\5C", '"' => "\\22")

Base.show(io::IO, o::ConcreteSyncScopes) = print(io, UnsafeAtomics, '.', llvm_syncscope(o))
Base.show(io::IO, o::LLVMSyncScope) =
    print(io, UnsafeAtomics, ".SyncScope(", repr(llvm_syncscope(o)), ')')

# The name of the scope to emit (see `native_scope`), as a `Val` for the generator. The
# canonical scopes can also be passed as a `Symbol`. Like orderings, scopes have to be
# constants.
@inline scope_val(scope::LLVMSyncScope) = Val(native_scope(llvm_syncscope(scope)))
@inline scope_val(scope::Symbol) =
    (scope === :system || scope === :device || scope === :workgroup || scope === :subgroup ||
     scope === :singlethread) ? Val(native_scope(scope)) : throw_invalid_scope()
scope_val(@nospecialize(scope)) = throw_invalid_scope()

# One literal: building the message at run time would allocate a string, which GPU code can't.
@noinline throw_invalid_scope() = throw(ArgumentError(
    "invalid syncscope: expected an UnsafeAtomics.SyncScope, or one of :singlethread, \
     :subgroup, :workgroup, :device or :system"))
