using GPJ, Test, Random

Random.seed!(123)

@testset "kernels" begin
    @testset "ZeroKernel" begin
        @test ZeroKernel{Float16}()(randn(3), randn(3)) == zeros(3)
        @test ZeroKernel{Float16}()(randn(3), randn(3); type=:pairwise) == zeros(3,3)
    end
end