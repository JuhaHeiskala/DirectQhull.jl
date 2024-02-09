using Test
using DirectQhull

@testset "ConvexHull Tests" begin
    points = [0 0 1 1 0.5 0.5 0.5; 0 1 0 1 0.5 0.3 0.7]  # Points that form a square with additional points in the interior
    hull = ConvexHull(points)

    @test all([v in 1:size(points, 2) for v in hull.vertices])
    expected_vertices = [1, 2, 3, 4]  # Indices of the corners of the square
    @test sort(hull.vertices) == sort(expected_vertices)
end
