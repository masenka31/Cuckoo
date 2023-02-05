using Cuckoo
using JsonGrinder
using Test
using DrWatson
@quickactivate

@testset "Paths" begin
    @test cuckoo_path == "/mnt/data/jsonlearning/Avast_cockoo"
    @test cuckoo_full_path == "/mnt/data/jsonlearning/Avast_cuckoo_full"
    @test benign_path == "/mnt/data/jsonlearning/garcia/reports/benign"
    @test malicious_path == "/mnt/data/jsonlearning/garcia/reports/malicious"

    @test expdir("test") == "/mnt/data/jsonlearning/experiments/test"
    @test resultsdir("test") == "/mnt/data/jsonlearning/results/test"
    @test splitsdir("test") == "/mnt/data/jsonlearning/splits/test"
end

@testset "Dataset()" begin
    cuckoo = Dataset("cuckoo", full=false)
    @test length(cuckoo.samples) == 48976

    cuckoo_full = Dataset("cuckoo", full=true)
    @test length(cuckoo_full.samples) == 48976

    garcia = Dataset("garcia")
    @test length(garcia.samples) == 19999
end