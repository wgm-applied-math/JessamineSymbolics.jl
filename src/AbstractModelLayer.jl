export model_basic_symbolic_output, model_symbolic_output

"""
    model_basic_symbolic_output(g_spec, genome, parameter, model_result; kw_args...)

Build a Symbolics form for the output of the final time step of
running `genome`
Then use the `parameter` vector,
the symbolic output of the genome,
and feed the symbolic output of the genome as input to
model result in `agent.extra` to make a prediction
in symbolic form.
No simplifications are performed.

The `kw_args` are eventually splatted into `model_predict`.

Return a named tuple with lots of useful fields.

"""
function model_basic_symbolic_output(
        g_spec::GenomeSpec,
        genome::AbstractGenome,
        parameter::AbstractArray,
        mr::AbstractModelResult;
        kw_args...)
    p, x, z = run_genome_symbolic(g_spec, genome)
    p_subs = Dict(p[j] => parameter[j] for j in eachindex(p))
    z_num = map(z) do zj
        substitute(zj, p_subs)
    end
    try
        z_sym_row_mat = reshape(z, 1, :)
        y_sym = model_predict(mr, z_sym_row_mat; kw_args...)[1]
        y_sub = substitute(y_sym, p_subs)
        y_num = Symbolics.simplify(y_sub)

        return (p = p, x = x, z = z, p_subs = p_subs, z_num = z_num,
            y_sym = y_sym,
            y_sub = y_sub, y_num = y_num)
    catch e
        @info "model_predict failed, returning short symbolic report: $(e)"
        return (p = p, x = x, z = z, p_subs = p_subs, z_num = z_num)
    end
end

"""
    model_basic_symbolic_output(g_spec, agent; kw_args...)

Build a Symbolics form for the output of the final time step of
running `agent`'s `genome`
Then use the `agent`'s `parameter` vector,
and feed the symbolic output of the genome as input to
model result in `agent.extra` to make a prediction
in symbolic form.

The `kw_args` are eventually splatted into `model_predict`.
"""
function model_basic_symbolic_output(g_spec, agent; kw_args...)
    return model_basic_symbolic_output(
        g_spec, agent.genome, agent.parameter, agent.extra; kw_args...)
end

"""
    model_symbolic_output(g_spec, agent; kw_args...)

Build a Symbolics form for the output of the final time step of
running `agent`'s `genome`
Then use the `agent`'s `parameter` vector,
and feed the symbolic output of the genome as input to
model result in `agent.extra` to make a prediction
in symbolic form.
The output tuple includes the result of various simplifications.

The `kw_args` are eventually splatted into `model_predict`.
"""
function model_symbolic_output(g_spec, agent; kw_args...)
    return model_symbolic_output(
        g_spec, agent.genome, agent.parameter, agent.extra; kw_args...)
end

"""
    model_symbolic_output(g_spec, genome, parameter, model_result; kw_args...)

Build a Symbolics form for the output of the final time step of
running `genome`
Then use the `parameter` vector,
the symbolic output of the genome,
and feed the symbolic output of the genome as input to
model result in `agent.extra` to make a prediction
in symbolic form.
The output tuple includes the result of various simplifications.

The `kw_args` are eventually splatted into `model_predict`.

Return a named tuple with lots of useful fields.

"""
function model_symbolic_output(
        g_spec::GenomeSpec,
        genome::AbstractGenome,
        parameter::AbstractArray,
        mr::AbstractModelResult;
        kw_args...)
    p, x, z = run_genome_symbolic(g_spec, genome)
    p_subs = Dict(p[j] => parameter[j] for j in eachindex(p))
    z_num = map(z) do zj
        substitute(zj, p_subs)
    end
    try
        z_sym_row_mat = reshape(z, 1, :)
        y_sym = model_predict(mr, z_sym_row_mat; kw_args...)[1]
        used_vars = Set(v.name for v in Symbolics.get_variables(y_sym))
        # To handle rational functions that have things like 1/(x/0),
        # replace Inf with W and do a limit as W -> Inf.
        # First, grind through and make sure we have a unique symbol.
        j = 0
        local W
        while true
            W = Symbolics.variable(:W, j)
            if !(W in used_vars)
                break
            end
            j += 1
        end
        y_W = substitute(y_sym, Dict([Inf => W]))
        y_lim = Symbolics.limit(y_W.val, W.val, Inf)
        y_simp = Symbolics.simplify(y_lim)
        y_sub = substitute(y_simp, p_subs)
        y_num = Symbolics.simplify(y_sub)

        return (p = p, x = x, z = z, p_subs = p_subs, z_num = z_num,
            y_sym = y_sym,
            y_lim = y_lim, y_simp = y_simp, y_sub = y_sub, y_num = y_num)
    catch e
        @info "model_predict failed, returning short symbolic report: $(e)"
        return (p = p, x = x, z = z, p_subs = p_subs, z_num = z_num)
    end
end
