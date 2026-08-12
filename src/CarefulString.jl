# This does floating point printout
using Base.Ryu

export careful_string
export CarefulStringStyle, JuliaStyle, PythonStyle

function dump_tree(expr, depth = 0)
    indent = " "^depth
    etype = typeof(expr)
    s = string(expr)
    println("$indent expr is of type $etype: $s")
    if iscall(expr)
        println("$indent is call")
        op = operation(expr)
        args = arguments(expr)
        println("$indent apply $op:")
        dump_tree(op, depth+1)
        println("$indent to args:")
        for arg in args
            dump_tree(arg, depth+1)
        end
    elseif isexpr(expr)
        println("$indent is expr")
        println("$indent head:")
        dump_tree(head(expr), depth+1)
        println("$indent children:")
        for c in children(expr)
            dump_tree(c, depth+1)
        end
    end
end

abstract type CarefulStringStyle end

struct PythonStyle <: CarefulStringStyle end

struct JuliaStyle <: CarefulStringStyle end

function careful_string(expr, style::CarefulStringStyle = JuliaStyle())
    io = IOBuffer()
    careful_string(io, expr, style)
    return String(take!(io))
end

# This is a fallback that should never be reached
function careful_string(io::IO, expr, style::CarefulStringStyle = JuliaStyle())
    etype = typeof(expr)
    s = string(expr)
    println("expr is of type $etype: $s")
    if iscall(expr)
        println("is call")
        op = operation(expr)
        args = arguments(expr)
        careful_string(io, (op, args), style)
    elseif isexpr(expr)
        println("Got expr")
        dump(expr)
    end
end

const BinOp = Union{typeof(+),typeof(-),typeof(*),typeof(/),typeof(//),typeof(^)}

function careful_string(io::IO, p::Tuple{BinOp,Any}, style::CarefulStringStyle = JuliaStyle())
    (op, args) = p
    @assert length(args) > 0
    if length(args) == 1
        careful_string(io, op, style)
        careful_string(io, args[1], style)
    elseif length(args) == 2
        print(io, "(")
        careful_string(io, args[1], style)
        print(io, " ")
        careful_string(io, op, style)
        print(io, " ")
        careful_string(io, args[2], style)
        print(io, ")")
    elseif op == (+) || op == (*)
        print(io, "(")
        careful_string(io, args[1], style)
        print(io, " ")
        careful_string(io, op, style)
        print(io, " ")
        careful_string(io, (op, args[2:end]), style)
        print(io, ")")
    else
        error("Not sure what to do with {op} and {length(args)} args")
    end
end

function careful_string(io::IO, p::Tuple{Function,Any}, style::CarefulStringStyle = JuliaStyle())
    (f, args) = p
    @assert length(args) > 0
    careful_string(io, f, style)
    print(io, "(")
    for j = 1:(length(args)-1)
        careful_string(io, args[j], style)
        print(io, ",")
    end
    careful_string(io, args[end], style)
    print(io, ")")
end

function careful_string(io::IO, f::Function, style::CarefulStringStyle = JuliaStyle())
    print(io, f)
end

function careful_string(io::IO, p::typeof(^), style::CarefulStringStyle = JuliaStyle())
    print(io, "^")
end

# For sympy
function careful_string(io::IO, p::typeof(^), style::PythonStyle)
    print(io, "**")
end

function careful_string(io::IO, p::typeof(//), style::CarefulStringStyle = JuliaStyle())
    print(io, "//")
end

# For sympy
function careful_string(io::IO, p::typeof(//), style::PythonStyle)
    print(io, "/")
end

function careful_string(io::IO, p::typeof(abs), style::CarefulStringStyle = JuliaStyle())
    print(io, "abs")
end

# For sympy
function careful_string(io::IO, p::typeof(abs), style::PythonStyle)
    print(io, "Abs")
end

function careful_string(io::IO, p::typeof(mod), style::CarefulStringStyle = JuliaStyle())
    print(io, "mod")
end

# For sympy
function careful_string(io::IO, p::typeof(mod), style::PythonStyle)
    print(io, "Mod")
end

function careful_string(io::IO, p::typeof(min), style::CarefulStringStyle = JuliaStyle())
    print(io, "min")
end

# For sympy
function careful_string(io::IO, p::typeof(min), style::PythonStyle)
    print(io, "Min")
end

function careful_string(io::IO, p::typeof(max), style::CarefulStringStyle = JuliaStyle())
    print(io, "max")
end

# For sympy
function careful_string(io::IO, p::typeof(max), style::PythonStyle)
    print(io, "Max")
end

function careful_string(io::IO, x::AbstractFloat, style::CarefulStringStyle = JuliaStyle())
    if isnan(x)
        print(io, "NaN")
    elseif isinf(x)
        if x > 0
            print(io, "Inf")
        else
            print(io, "(-Inf)")
        end
    else
        basic = Ryu.writeshortest(Float64(x))
        partsrx = r"(?<m>-?\d+(\.\d*)?)([eE](?<e>[+-]?\d+))?"
        m = match(partsrx, basic)
        @assert m isa RegexMatch "Unable to parse real: $x as '$basic'"
        me = m["e"]
        mm = m["m"]
        if !isnothing(me)
            print(io, "($mm*10")
            careful_string(io, ^, style)
            print(io, "$me)")
        else
            print(io, basic)
        end
    end
end

function careful_string(io::IO, x::Rational, style::CarefulStringStyle = JuliaStyle())
    p = numerator(x)
    q = denominator(x)
    if q == 0
        print(io, "($p/ε)")
    else
        print(io, "($p/$q)")
    end
end

function careful_string(io::IO, x::Integer, style::CarefulStringStyle = JuliaStyle())
    print(io, x)
end

function careful_string(io::IO, x::Number, style::CarefulStringStyle = JuliaStyle())
    print(io, x)
end

function careful_string(io::IO, expr::Num, style::CarefulStringStyle = JuliaStyle())
    v = Symbolics.unwrap(expr)
    careful_string(io, v, style)
end

function careful_string(io::IO, expr::SymbolicUtils.BasicSymbolic, style::CarefulStringStyle = JuliaStyle())
    if SymbolicUtils.issym(expr)
        print(io, string(expr))
    elseif SymbolicUtils.isconst(expr)
        careful_string(io, SymbolicUtils.unwrap_const(expr), style)
    elseif iscall(expr)
        op = operation(expr)
        args = arguments(expr)
        careful_string(io, (op, args), style)
    elseif isexpr(expr)
        println("Got expr")
        dump(expr)
    else
        println(io, "basic symbolic: $expr")
        dump(expr)
    end
end

function careful_string(io::IO, expr::SymbolicUtils.BasicSymbolic, style::PythonStyle)
    if SymbolicUtils.issym(expr)
        raw = string(expr)
        fixed = replace_subscripts(raw)
        print(io, fixed)
    elseif SymbolicUtils.isconst(expr)
        careful_string(io, SymbolicUtils.unwrap_const(expr), style)
    elseif iscall(expr)
        op = operation(expr)
        args = arguments(expr)
        careful_string(io, (op, args), style)
    elseif isexpr(expr)
        println("Got expr")
        dump(expr)
    else
        println(io, "basic symbolic: $expr")
        dump(expr)
    end
end


const SUBSCRIPTS = [
    '\u2080' => '0',   # ₀
    '\u2081' => '1',   # ₁
    '\u2082' => '2',   # ₂
    '\u2083' => '3',   # ₃
    '\u2084' => '4',   # ₄
    '\u2085' => '5',   # ₅
    '\u2086' => '6',   # ₆
    '\u2087' => '7',   # ₇
    '\u2088' => '8',   # ₈
    '\u2089' => '9',   # ₉
]

function replace_subscripts(s)
    replace(s, SUBSCRIPTS...)
end
