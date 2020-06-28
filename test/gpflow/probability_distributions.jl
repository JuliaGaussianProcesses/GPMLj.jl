using GPMLj, Test, Random, PyCall

Random.seed!(123)

@testset "ProbabilityDistributions" begin

    @testset "Gaussian" begin
        prior = gpflow.probability_distributions.Gaussian(0.0, 1.0)
        typeof(prior)<:gpflow.AbstractProbabilityDistribution
        @test typeof(prior)<:gpflow.probability_distributions.Gaussian
        @test typeof(prior.o)<:PyObject
        temp = prior.o
        @test instantiate!(prior) == temp
    end

    @testset "DiagonalGaussian" begin
        prior = gpflow.probability_distributions.DiagonalGaussian(0.0, 1.0)
        typeof(prior)<:gpflow.AbstractProbabilityDistribution
        @test typeof(prior)<:gpflow.probability_distributions.DiagonalGaussian
        @test typeof(prior.o)<:PyObject
        temp = prior.o
        @test instantiate!(prior) == temp
    end

    @testset "MarkovGaussian" begin
        prior = gpflow.probability_distributions.MarkovGaussian(0.0, 1.0)
        typeof(prior)<:gpflow.AbstractProbabilityDistribution
        @test typeof(prior)<:gpflow.probability_distributions.MarkovGaussian
        @test typeof(prior.o)<:PyObject
        temp = prior.o
        @test instantiate!(prior) == temp
    end
end
