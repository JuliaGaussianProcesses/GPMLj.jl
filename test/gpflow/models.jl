using GPMLj, Test, Random, PyCall

Random.seed!(123)


@testset "Models" begin
    @testset "GPR" begin
        kernel = gpflow.kernels.Matern52()
        gpr = gpflow.models.GPR((randn(100,2), randn(100,1)), kernel)
        @test typeof(gpr)<:gpflow.models.GPR
        @test typeof(gpr)<:gpflow.Model
        @test typeof(gpr.o)<:PyObject
        @test typeof(gpr.kernel.o)<:PyObject

    end

    # FIXME
    #= @testset "SGPR" begin
        kern= gpflow.kernels.Matern52(2)
        sgpr = gpflow.models.SGPR(randn(100,2), randn(100,1), kern;)
        @test typeof(sgpr)<:gpflow.models.SGPR
        @test typeof(sgpr)<:gpflow.Model
        @test typeof(sgpr.o)<:PyObject
        @test typeof(sgpr.kern.o)<:PyObject

    end =#
    
    @testset "VGP" begin
        kernel = gpflow.kernels.Matern52()
        like = gpflow.likelihoods.Gaussian()
        vgp = gpflow.models.VGP((randn(100,2), randn(100,1)), kernel, like)
        @test typeof(vgp)<:gpflow.models.VGP
        @test typeof(vgp)<:gpflow.Model
        @test typeof(vgp.o)<:PyObject
        @test typeof(vgp.kernel.o)<:PyObject
        @test typeof(vgp.likelihood.o)<:PyObject
    end

    @testset "GPMC" begin
        kernel = gpflow.kernels.Matern52()
        like = gpflow.likelihoods.Gaussian()
        gpmc = gpflow.models.GPMC((randn(100,2), randn(100,1)), kernel, like)
        @test typeof(gpmc)<:gpflow.models.GPMC
        @test typeof(gpmc)<:gpflow.Model
        @test typeof(gpmc.o)<:PyObject
        @test typeof(gpmc.kernel.o)<:PyObject
        @test typeof(gpmc.likelihood.o)<:PyObject
    end 

end #module
