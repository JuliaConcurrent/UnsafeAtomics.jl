# The `public` keyword doesn't parse before Julia 1.11.
function declare_public(mod::Module, names::Symbol...)
    @static if VERSION >= v"1.11.0-DEV.469"
        Core.eval(mod, Expr(:public, names...))
    end
    return
end
