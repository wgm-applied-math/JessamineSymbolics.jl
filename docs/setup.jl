using Pkg
Pkg.develop(PackageSpec(url="https://github.com/wgm-applied-math/Jessamine.jl.git", rev="v0.4.0"))
Pkg.develop(PackageSpec(path=pwd()))
Pkg.instantiate()
