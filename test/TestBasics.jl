module TestBasics
using Random
using Symbolics
using Test
using Jessamine
using JessamineSymbolics

include("RandomData.jl")

g_spec = GenomeSpec(4, 0, 1, 2, 3)
index_max = workspace_size(g_spec)
lambda_b = 1e-6
lambda_p = 1e-6
lambda_o = 1e-6

# Adapt least_squares_ridge_grow_and_rate so that it can be used by next_generation
function grow_and_rate(rng, g_spec, genome)
    return least_squares_ridge_grow_and_rate(
        RD.xs, RD.y, lambda_b, lambda_p, lambda_o,
        g_spec, genome)
end

# Indices into the state vector
z1, z2, z3, z4, p1, x1, x2 = 1:index_max

# This should output [1, x1, x2, x1 * x2]
g_check = Genome(
    [[Instruction(Multiply(), Int[])], # creates a 1
     [Instruction(Multiply(), [x1])],
     [Instruction(Multiply(), [x2])],
     [Instruction(Multiply(), [x1, x2])]])

# This should result in 2 + 3 x1 + 3 x2 - 3 x1 * x2
a_check = grow_and_rate(Random.default_rng, g_spec, g_check)
@show a_check
short_show(a_check)
@show round.(coefficients(a_check.extra))
@test round.(coefficients(a_check.extra)) == [2, 3, 3, -3]
@test intercept(a_check.extra) == 0

basic_sym_res = model_basic_symbolic_output(g_spec, a_check)
sym_res = model_symbolic_output(g_spec, a_check)
x = sym_res.x

@show basic_sym_res.y_sym
@show basic_sym_res.y_num
@show sym_res.y_sym
@show a_check.parameter
@show a_check.extra
@show sym_res.y_num

y_num = simplify(sym_res.y_num; expand = true)
@show y_num

c1 = Symbolics.coeff(y_num)
cx1 = Symbolics.coeff(Symbolics.coeff(y_num, x[1]))
cx2 = Symbolics.coeff(Symbolics.coeff(y_num, x[2]))
cx1x2 = Symbolics.coeff(Symbolics.coeff(y_num, x[2]), x[1])
cvec = [c1, cx1, cx2, cx1x2]
@show cvec
@show round.(cvec)
@test round.(cvec) == [2, 3, 3, -3]

y_rni = replace_near_integer(y_num, tolerance=1e-6)
@show y_rni

end
