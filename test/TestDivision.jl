module TestDivision

using LinearAlgebra
using Random
using Symbolics
using Test

using Jessamine
using JessamineSymbolics

# Here's the core problem:
#
# using Symbolics
# zr = Num(1) / Num(0)
#
# That results in zr = Num(1 // 0)
# Now when you do this:
#
# 0 * zr
#
# The result is a call to Julia's built-ins 0 * (1//0)
# which throws an integer division error.
# Oddly, if you do this instead:
#
# @variables x y z
#
# substitute(x*zr, Dict(x => 0)) # error
# substitute(x/y, Dict(x => 0, y => 0)) # Yields 1.
# substitute(x*y, Dict(x => 0, y => zr)) # error

# The problem actually seems to be that when computing the
# symbolic form of a genome, the early stages can trigger this
# error, but they would be resolved later because the final form
# does not have any 0/0s.  That is, maybe a scratch variable (t)
# is set to 0/0 but is not used when computing an output value
# (z).  A floating-point calculation with that genome results in
# some Inf and NaN, but since they only show up in scratch
# calculations, the genome as a whole is successful.

# Set up a genome that eventually does 0 times 1//0

g_spec = GenomeSpec(1, 2, 1, 1, 3)
index_max = workspace_size(g_spec)

z1, t1, t2, p1, x1 = 1:index_max

# z1 = t1 * t2 + t1
# t1 = 1/t2
# t2 = 0 # A literal integer zero.
g_check = Genome(
    [[Instruction(Multiply(), [t1, t2])],
     [Instruction(ReciprocalAdd(), [t2])],
     []]
)

short_show(stdout, g_check)

@show run_genome_to_last(g_spec, g_check, [-6.2], [0.0])

println(show_symbolic(g_spec, g_check))

a = Agent(nothing, g_check, [-6.2], nothing)

s_run = run_genome_symbolic(g_spec, g_check)
@show s_run

s_form = symbolic_form(g_spec, a)
@show s_form

end
