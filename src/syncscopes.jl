struct LLVMSyncScope{name} <: SyncScope end

const none = LLVMSyncScope{:none}()
const singlethread = LLVMSyncScope{:singlethread}()
const subgroup = LLVMSyncScope{:subgroup}()
const workgroup = LLVMSyncScope{:workgroup}()
const device = LLVMSyncScope{:device}()
const system = none

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

# The name of the syncscope for the generator, which calls the default scope `:system`.
scope_name(s::LLVMSyncScope) = s === system ? :system : llvm_syncscope(s)

# Call `f(Val(name), args...)` with the name of the scope. Like `with_ordering`, this branches
# on the value, so a scope that isn't a constant is a branch per canonical scope, not a dynamic
# call. The canonical scopes can also be passed as a `Symbol`. Other scopes have to be known
# from their type: Julia widens a `Union` of more than 3 scopes to the abstract type, which
# leaves a dynamic call for them, so scopes chosen at run time are best passed as Symbols.
@inline with_scope(f, scope, args...) =
    (scope === system || scope === :system) ? f(Val(:system), args...) :
    (scope === device || scope === :device) ? f(Val(:device), args...) :
    (scope === workgroup || scope === :workgroup) ? f(Val(:workgroup), args...) :
    (scope === subgroup || scope === :subgroup) ? f(Val(:subgroup), args...) :
    (scope === singlethread || scope === :singlethread) ? f(Val(:singlethread), args...) :
    scope isa LLVMSyncScope ? f(Val(scope_name(scope)), args...) :
    throw_invalid_scope()

# `f(order, scope)` and `f(success_order, failure_order, scope)` with `Val`s of the names.
# Orderings and scopes that aren't constants are passed along as arguments, never captured in
# a closure: the type of such a closure depends on their run-time type, which would make
# creating it a dynamic call.
@inline with_ordering_and_scope(f, order, scope) =
    with_ordering(_with_scope, order, f, scope)
@inline _with_scope(o, f, scope) = with_scope(_call_with, scope, f, o)
@inline _call_with(s, f, o) = f(o, s)

@inline with_orderings_and_scope(f, success, failure, scope) =
    with_ordering(_with_failure, success, f, failure, scope)
@inline _with_failure(so, f, failure, scope) =
    with_ordering(_with_scope, failure, (fo, s) -> f(so, fo, s), scope)

@noinline throw_invalid_scope() = throw(ArgumentError(
    "invalid syncscope: expected an UnsafeAtomics.SyncScope, or one of :singlethread, " *
    ":subgroup, :workgroup, :device or :system"))
