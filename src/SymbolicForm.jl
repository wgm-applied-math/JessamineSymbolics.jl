export eval_time_step_symbolic, show_symbolic, run_genome_symbolic
export replace_near_integer
export compile_to_function, symbolic_form

"""
    eval_time_step_symbolic(g_spec, genome; output_sym, scratch_sym, parameter_sym, input_sym)

Return symbolic objects representing a single time step of `genome`.
The result is a named tuple with fields
`z`, `t`, `p`, `x` for the `Symbolics` objects for those variables;
`c` and `c_next` for the current and future cell state in symbolic form.
"""
function eval_time_step_symbolic(g_spec::GenomeSpec, genome::Genome;
        output_sym = :z,
        scratch_sym = :t,
        parameter_sym = :p,
        input_sym = :x)
    z = Symbolics.variables(output_sym, 1:(g_spec.output_size))
    t = Symbolics.variables(scratch_sym, 1:(g_spec.scratch_size))
    p = Symbolics.variables(parameter_sym, 1:(g_spec.parameter_size))
    x = Symbolics.variables(input_sym, 1:(g_spec.input_size))
    c = CellState(z, t, p, x)
    c_next = eval_time_step(c, genome)
    return (z = z, t = t, p = p, x = x, c = c, c_next = c_next)
end

"""
    show_symbolic(g_spec, genome; output_sym, scratch_sym, parameter_sym, input_sym)

Return a matrix with three columns.
The first is row indices 1, 2, etc.
The second is the initial workspace vector in symbolic form.
The third is the future workspace state in symbolic form.
"""
function show_symbolic(g_spec::GenomeSpec, genome::Genome;
        output_sym = :z,
        scratch_sym = :t,
        parameter_sym = :p,
        input_sym = :x)
    z, t, p, x, c, c_next = eval_time_step_symbolic(g_spec, genome;
        output_sym = output_sym,
        scratch_sym = scratch_sym,
        parameter_sym = parameter_sym,
        input_sym = input_sym)
    return hcat(1:length(c), flat_workspace(c), flat_workspace(c_next))
end

"""
    run_genome_symbolic(g_spec, genome; parameter_sym=:p, input_sym=:x)

Build a symbolic form for the output of the final time step of
running `genome`.  The parameter vector and input vector are
`Symbolics` objects of the form `p[j]` and `x[j]`.  The variable
names can be specified with the keyword arguments.

Return a named tuple `(p, x, z)` where `p` and `x`, are vectors
of `Symbolics` objects used to represent genome parameters and
inputs; and `z` is a vector of genome outputs in symbolic form,
equivalent to `run_genome_to_last`.
"""
function run_genome_symbolic(
        g_spec::GenomeSpec,
        genome::AbstractGenome;
        parameter_sym = :p,
        input_sym = :x)
    p = Symbolics.variables(parameter_sym, 1:(g_spec.parameter_size))
    x = Symbolics.variables(input_sym, 1:(g_spec.input_size))

    z = run_genome_to_last(g_spec, genome, p, x)
    return (p = p, x = x, z = z)
end

"""
    symbolic_form(g_spec::GenomeSpec, agent::Agent)

Produce a symbolic form for an agent.  Return a named tuple with
field `x` equal to a symbol for the input array, and field `vf`
equal to the outputs expressed in terms of `x`.
"""
function symbolic_form(g_spec::GenomeSpec, agent::Agent)
    genome_sym = run_genome_symbolic(g_spec, agent.genome)
    p_subs = Dict(genome_sym.p[j] => agent.parameter[j] for j in eachindex(genome_sym.p))
    a_sym = substitute(genome_sym.z, p_subs)
    return (x = genome_sym.x, vf = a_sym)
end

"""
    replace_near_integer(expr; tolerance=1.0e-10)

Round literal floating-point numbers within a symbolic expression.
Specifically, if a number `a` differs `round(a)` by less thant `tolerance`,
it gets replaced by `round(a)`.
"""
function replace_near_integer(expr::SymbolicUtils.BasicSymbolic; tolerance = 1.0e-10)
    if TermInterface.iscall(expr)
        op = TermInterface.operation(expr)
        args = TermInterface.arguments(expr)
        # For some reason, the wrap() here is necessary
        # to get literal numbers into a subtype of Num,
        # which can then trigger the correct recursive descent
        # down to actual rounding
        new_args = map(arg -> replace_near_integer(Symbolics.wrap(arg), tolerance = tolerance), args)
        return TermInterface.maketerm(typeof(expr), op, new_args, TermInterface.metadata(expr))
    else  # This should handle symbols and other atomic expressions
        return expr
    end
end

function replace_near_integer(expr::Num; tolerance = 1.0e-10)
    # Originally used Symbolics.value, but that may
    # be deprecated. Symbolics.symbolic_to_float is currently documented.
    return Num(replace_near_integer(Symbolics.symbolic_to_float(expr); tolerance = tolerance))
end

function replace_near_integer(expr::Number; tolerance = 1.0e-10)
    k = round(expr)
    if k < typemax(Int) && abs(expr - k) < tolerance
        return convert(Num, convert(Int, k))
    else
        return expr
    end
end


"""
    DOTMAP

(private) A mapping from standard functions to the symbols for their element-wise (dot) counterparts.
"""
const DOTMAP = Dict(
    (+) => :(.+),
    (-) => :(.-),
    (*) => :(.*),
    (/) => :(./),
    (^) => :(.^),
)

"""
    insert_dot(e)

(private) Recursively traverses the expression `e`, replacing standard functions with
the symbols for their element-wise (dot) counterparts according to `DOTMAP`.
"""
function insert_dot(e)
    if isa(e, Symbol)
        if e in [:+, :-, :*, :/, :^]
            return Symbol("." * string(e))
        else
            return e
        end
    elseif isa(e, Function)
        if e in keys(DOTMAP)
            return DOTMAP[e]
        else
            return e
        end
    elseif isa(e, Expr)
        new_args = [insert_dot(arg) for arg in e.args]
        return Expr(e.head, new_args...)
    else
        return e
    end
end

"""
    compile_to_function(g_spec::GenomeSpec, genome::AbstractGenome)

Compile a genome to a function.  The returned value is a callable
object that takes a vector of inputs and a vector of parameters,
and produces an array of outputs, equivalent to calling
`run_genome_to_last`.
"""


"""
    compile_to_function(g_spec::GenomeSpec, agent::Agent)

Compile an agent to a function.  The returned value is a callable
object that takes the x's as inputs and produces an output, using
the parameter vector embedded in `agent`.  The function is
compiled within the `JessamineSymbolics` module and should
generally be used with dot broadcasting when called on inputs
that are vectors.
"""
function compile_to_function(g_spec::GenomeSpec, agent::Agent)
    basic_sym_res = model_basic_symbolic_output(g_spec, agent)
    y_expr = SymbolicUtils.Code.toexpr(basic_sym_res.y_num)
    x_expr = SymbolicUtils.Code.toexpr.(basic_sym_res.x)
    fn_expr = :(
        $(Expr(:tuple, x_expr...)) -> $y_expr
    )
    @show fn_expr
    return @RuntimeGeneratedFunction(fn_expr)
end
