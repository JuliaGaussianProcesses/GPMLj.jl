using GPMLj, Test, Random, PyCall

Random.seed!(123)


@testset "Kernels" begin
    @testset "Matern12" begin
        kern = gpflow.kernels.Matern12()
        @test typeof(kern)<:gpflow.Kernel
        @test typeof(kern)<:gpflow.kernels.AbstractStationary
        @test typeof(kern)<:gpflow.kernels.Matern12
        @test typeof(kern.o)<:PyObject
        temp = kern.o
        @test instantiate!(kern) == temp
    end    

    @testset "Matern32" begin
        kern = gpflow.kernels.Matern32()
        @test typeof(kern)<:gpflow.Kernel
        @test typeof(kern)<:gpflow.kernels.AbstractStationary
        @test typeof(kern)<:gpflow.kernels.Matern32
        @test typeof(kern.o)<:PyObject
        temp = kern.o
        @test instantiate!(kern) == temp
    end

    @testset "Matern52" begin
        kern = gpflow.kernels.Matern52()
        @test typeof(kern)<:gpflow.Kernel
        @test typeof(kern)<:gpflow.kernels.AbstractStationary
        @test typeof(kern)<:gpflow.kernels.Matern52
        @test typeof(kern.o)<:PyObject
        temp = kern.o
        @test instantiate!(kern) == temp
    end

    # @testset "ArcCosine" begin
    #     kern = gpflow.kernels.ArcCosine(2)
    #     @test typeof(kern)<:gpflow.Kernel
    #     @test typeof(kern)<:gpflow.kernels.ArcCosine
    #     @test typeof(kern.o)<:PyObject
    #     temp = kern.o
    #     @test instantiate!(kern) == temp
    # end

    # @testset "Periodic" begin
    #     kern = gpflow.kernels.Periodic(2)
    #     @test typeof(kern)<:gpflow.Kernel
    #     @test typeof(kern)<:gpflow.kernels.Periodic
    #     @test typeof(kern.o)<:PyObject
    #     temp = kern.o
    #     @test instantiate!(kern) == temp
    # end

    # # TODO
    # # @testset "Coregion" begin
    # #     kern= gpflow.kernels.Coregion(2, 1)
    # #     @test typeof(kern)<:gpflow.Kernel
    # #     @test typeof(kern)<:gpflow.kernels.Coregion
    # #     @test typeof(kern.o)<:PyObject
    # #     temp = kern.o
    # #     @test instantiate!(kern) == temp
    # # end


    # # Static Kernels
    # @testset "Static" begin
    #     kern = gpflow.kernels.Static()
    #     @test typeof(kern)<:gpflow.Kernel
    #     @test typeof(kern)<:gpflow.kernels.AbstractStatic
    #     @test typeof(kern)<:gpflow.kernels.Static
    #     @test typeof(kern.o)<:PyObject
    #     temp = kern.o
    #     @test instantiate!(kern) == temp
    # end

    # @testset "Combination" begin
    #     kern1 = gpflow.kernels.Matern52(2)
    #     kern2 = gpflow.kernels.ArcCosine(2)
    #     kern3 = gpflow.kernels.Periodic(2)
    #     kernels = [kern1, kern2, kern3];
    #     kern = gpflow.kernels.Combination(kernels)
    #     @test typeof(kern)<:gpflow.Kernel
    #     @test typeof(kern)<:gpflow.kernels.AbstractCombination
    #     @test typeof(kern)<:gpflow.kernels.Combination
    #     @test typeof(kern.o)<:PyObject
    #     temp = kern.o
    #     @test instantiate!(kern) == temp
    # end

    # @testset "Product" begin
    #     kern1 = gpflow.kernels.Matern52(2)
    #     kern2 = gpflow.kernels.ArcCosine(2)
    #     kern3 = gpflow.kernels.Periodic(2)
    #     kernels = [kern1, kern2, kern3];
    #     kern = gpflow.kernels.Product(kernels)
    #     @test typeof(kern)<:gpflow.Kernel
    #     @test typeof(kern)<:gpflow.kernels.AbstractCombination
    #     @test typeof(kern)<:gpflow.kernels.Product
    #     @test typeof(kern.o)<:PyObject
    #     temp = kern.o
    #     @test instantiate!(kern) == temp
    # end

    # @testset "Sum" begin
    #     kern1 = gpflow.kernels.Matern52(2)
    #     kern2 = gpflow.kernels.ArcCosine(2)
    #     kern3 = gpflow.kernels.Periodic(2)
    #     kernels = [kern1, kern2, kern3];
    #     kern = gpflow.kernels.Sum(kernels)
    #     @test typeof(kern)<:gpflow.Kernel
    #     @test typeof(kern)<:gpflow.kernels.AbstractCombination
    #     @test typeof(kern)<:gpflow.kernels.Sum
    #     @test typeof(kern.o)<:PyObject
    #     temp = kern.o
    #     @test instantiate!(kern) == temp
    # end

    # @testset "Convolutional" begin
    #     basekern = gpflow.kernels.Matern52(4)
    #     kern = gpflow.kernels.Convolutional(basekern, [3, 3], [2, 2])
    #     @test typeof(kern)<:gpflow.Kernel
    #     @test typeof(kern)<:gpflow.kernels.Convolutional
    #     @test typeof(kern.o)<:PyObject
    #     temp = kern.o
    #     @test instantiate!(kern) == temp
    # end

    # @testset "WeightedConvolutional" begin
    #     basekern = gpflow.kernels.Matern52(4)
    #     kern = gpflow.kernels.Convolutional(basekern, [3, 3], [2, 2])
    #     @test typeof(kern)<:gpflow.Kernel
    #     @test typeof(kern)<:gpflow.kernels.Convolutional
    #     @test typeof(kern.o)<:PyObject
    #     temp = kern.o
    #     @test instantiate!(kern) == temp
    # end

    # @testset "Cosine" begin
    #     kern = gpflow.kernels.Cosine(2)
    #     @test typeof(kern)<:gpflow.Kernel
    #     @test typeof(kern)<:gpflow.kernels.AbstractStationary
    #     @test typeof(kern)<:gpflow.kernels.Cosine
    #     @test typeof(kern.o)<:PyObject
    #     temp = kern.o
    #     @test instantiate!(kern) == temp
    # end

    # @testset "Exponential" begin
    #     kern = gpflow.kernels.Exponential(2)
    #     @test typeof(kern)<:gpflow.Kernel
    #     @test typeof(kern)<:gpflow.kernels.AbstractStationary
    #     @test typeof(kern)<:gpflow.kernels.Exponential
    #     @test typeof(kern.o)<:PyObject
    #     temp = kern.o
    #     @test instantiate!(kern) == temp
    # end

    # @testset "RationalQuadratic" begin
    #     kern = gpflow.kernels.RationalQuadratic(2)
    #     @test typeof(kern)<:gpflow.Kernel
    #     @test typeof(kern)<:gpflow.kernels.AbstractStationary
    #     @test typeof(kern)<:gpflow.kernels.RationalQuadratic
    #     @test typeof(kern.o)<:PyObject
    #     temp = kern.o
    #     @test instantiate!(kern) == temp
    # end

    # @testset "SquaredExponential" begin
    #     kern = gpflow.kernels.SquaredExponential(2)
    #     @test typeof(kern)<:gpflow.Kernel
    #     @test typeof(kern)<:gpflow.kernels.AbstractStationary
    #     @test typeof(kern)<:gpflow.kernels.SquaredExponential
    #     @test typeof(kern.o)<:PyObject
    #     temp = kern.o
    #     @test instantiate!(kern) == temp
    # end

    # @testset "Stationary" begin
    #     kern = gpflow.kernels.Stationary(2)
    #     @test typeof(kern)<:gpflow.Kernel
    #     @test typeof(kern)<:gpflow.kernels.AbstractStationary
    #     @test typeof(kern)<:gpflow.kernels.Stationary
    #     @test typeof(kern.o)<:PyObject
    #     temp = kern.o
    #     @test instantiate!(kern) == temp
    # end

    # @testset "Linear" begin
    #     kern = gpflow.kernels.Linear(2)
    #     @test typeof(kern)<:gpflow.Kernel
    #     @test typeof(kern)<:gpflow.kernels.AbstractLinear
    #     @test typeof(kern)<:gpflow.kernels.Linear
    #     @test typeof(kern.o)<:PyObject
    #     temp = kern.o
    #     @test instantiate!(kern) == temp
    # end

    # @testset "Polynomial" begin
    #     kern = gpflow.kernels.Polynomial(2)
    #     @test typeof(kern)<:gpflow.Kernel
    #     @test typeof(kern)<:gpflow.kernels.AbstractLinear
    #     @test typeof(kern)<:gpflow.kernels.Polynomial
    #     @test typeof(kern.o)<:PyObject
    #     temp = kern.o
    #     @test instantiate!(kern) == temp
    # end
end #module
