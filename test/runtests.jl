using JessamineSymbolics
using Test

@testset "JessamineSymbolics.jl" begin
    @testset "Basics" begin
        println("=== TestBasics ===")
        include("TestBasics.jl")
    end
    @testset "Division" begin
        println("=== TestDivision ===")
        include("TestDivision.jl")
    end
end
