using LinearAlgebra
using PlyIO
using BenchmarkTools

function load_ply_as_array(fname)
    bunny_ply = load_ply(fname)
    bunny_pcl = hcat(
       bunny_ply["vertex"]["x"],
        bunny_ply["vertex"]["y"],
        bunny_ply["vertex"]["z"]
    )
    bunny_pcl
end


function project_points(points, k::Matrix, dist)
    XYZ = points
    N = size(points)[1]
    u = Array{eltype(points)}(undef, N)
    v = Array{eltype(points)}(undef, N)

    fx = k[1, 1]
    fy = k[2, 2]
    cx = k[1, 3]
    cy = k[2, 3]
    k1, k2, p1, p2, k3 = dist

    for i=1:N
        x = XYZ[i, 1] / XYZ[i, 3]
        y = XYZ[i, 2] / XYZ[i, 3]

        r2 = x*x + y*y
        r4 = r2*r2
        r6 = r4*r2
        a1 = 2*x*y
        a2 = r2 + 2*x*x
        a3 = r2 + 2*y*y


        cdist = (1+k1*r2 + k2*r4 + k3*r6)
        xd = x * cdist + p1*a1 + p2*a2
        yd = y * cdist + p1*a3 + p2*a1

        u[i] =  xd * fx + cx
        v[i] =  yd * fy + cy
    end
    u, v
end


pcl = load_ply_as_array("../data/bun_zipper.ply")
w, h  = 640, 480
fx, fy = 500, 500
cx, cy = w/2, h/2
k = [
    fx 0 cx;
    0 fy cy;
    0 0 1
]
k1, k2, k3 = 0.09, -0.05, 0.03
p1, p2 = 0.06, 0.15
dist_coeff = [k1, k2, p1, p2, k3]

@benchmark uv = project_points(pcl, k, dist_coeff)

