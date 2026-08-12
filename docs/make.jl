using Documenter, DocumenterMarkdown
using JessamineSymbolics

makedocs(
    modules = [JessamineSymbolics],
    sitename = "Documentation for JessamineSymbolics"
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/wgm-applied-math/JessamineSymbolics.jl.git"
)
