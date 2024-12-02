struct LLVMSyncScope{name} <: SyncScope end

const none = LLVMSyncScope{Symbol("")}()
const singlethread = LLVMSyncScope{:singlethread}()

llvm_syncscope(::LLVMSyncScope{name}) where {name} = name

Base.string(s::LLVMSyncScope) = String(llvm_syncscope(s))
Base.print(io::IO, s::LLVMSyncScope) = print(io, string(s))
