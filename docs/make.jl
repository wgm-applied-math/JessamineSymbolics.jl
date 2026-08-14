using JessamineSymbolics
using Documenter

DocMeta.setdocmeta!(JessamineSymbolics, :DocTestSetup, :(using JessamineSymbolics); recursive=true)

makedocs(
    modules = [JessamineSymbolics],
    authors="W. Garrett Mitchener <mitchenerg@charleston.edu> and others",
    sitename="JessamineSymbolics.jl",
    format=Documenter.HTML(;
        canonical="https://wgm-applied-math.github.io/JessamineBenchmark.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/wgm-applied-math/JessamineSymbolics.jl.git",
    devbranch = "main",
    versions = ["stable" => "v^", "v#.#", "dev" =>  "dev"] # Explicitly forces version tracking
)
