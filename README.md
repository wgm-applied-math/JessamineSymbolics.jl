# JessamineSymbolics.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://wgm-applied-math.github.io/JessamineSymbolics.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://wgm-applied-math.github.io/JessamineSybolics.jl/dev/)
[![Build Status](https://github.com/wgm-applied-math/Jessamine.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/wgm-applied-math/JessamineSymbolics.jl/actions/workflows/CI.yml?query=branch%3Amain)

## About

Jessamine is a collection of [Julia](https://www.julialang.org) packages for machine learning, specifically, evolutionary symbolic regression and classification using static-single-assignment-form expressions.
It is a research project under development and not (yet) easy to use.
Expect ongoing improvements and breaking changes.

As of 2026-08-18, I have not registered this package.
To use this package within a Julia project, use the [Pkg.jl](https://pkgdocs.julialang.org/v1/) command line,
```julia-repl
pkg> add https://github.com/wgm-applied-math/Jessamine.jl#main
```
and you can use a tag or branch name in place of `main`.

This package extends the core [Jessamine.jl](https://github.com/wgm-applied-math/Jessamine.jl) package with capabilities from [Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl).

See the core package repository
[Jessamine.jl](https://github.com/wgm-applied-math/Jessamine.jl)
for more information.

## Installation

As of 2026-08-18, I have not registered this package.
To use this package within a Julia project, use the [Pkg.jl](https://pkgdocs.julialang.org/v1/) command line,
```julia-repl
pkg> add https://github.com/wgm-applied-math/JessamineSymbolics.jl#main
```
You can use a tag or branch name in place of `main`.

## Important note

As of 2026-08-18, there are several important bugs in the Symbolics.jl package that have yet to be resolved.
These don't seem to directly impact this package, but there are obvious features I can't add to this package until those bugs are resolved.
