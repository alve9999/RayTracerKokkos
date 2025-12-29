#include "BVH.hpp"
#include "random_scene.hpp"
#include "sampler.hpp"
#include "window.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <SDL2/SDL.h>
#include <chrono>
#include <cmath>
#include <decl/Kokkos_Declare_CUDA.hpp>
#include <iostream>
#include <vector>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

Assimp::Importer importer;

void display(Kokkos::View<Vec3 *, ExecSpace> colors, int width, int height,
             int current_spp, SDLHelper &sdl) {
    auto host = Kokkos::create_mirror_view(colors);
    Kokkos::deep_copy(host, colors);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            Vec3 c = host(idx) * (1.0f / float(current_spp));

            c.x = std::min(std::sqrt(c.x), 1.0f);
            c.y = std::min(std::sqrt(c.y), 1.0f);
            c.z = std::min(std::sqrt(c.z), 1.0f);

            sdl.write_pixel(x, y, c.x * 255, c.y * 255, c.z * 255);
        }
    }

    sdl.update();
}
const int width = 1920;
const int height = 1080;
const int spp = 10000;
using ExecSpace = Kokkos::Cuda;
int main(int argc, char *argv[]) {
    const aiScene *scene = importer.ReadFile(
        "../Untitled.obj", aiProcess_Triangulate | aiProcess_GenSmoothNormals |
                               aiProcess_JoinIdenticalVertices);

    if (!scene) {
        std::cerr << "Error: " << importer.GetErrorString() << "\n";
        return 1;
    }

    std::vector<cpuTriangle> triangles;
    for (int i = 0; i < scene->mNumMeshes; i++) {
        aiMesh *mesh = scene->mMeshes[i];
        gather_triangles(mesh, triangles);
    }
    std::vector<cpuTriangle> room = create_scene_geometry();
    std::vector<int> bvh_indices;
    std::vector<BVHNode> bvh_roots;
    int bvh_index = 0;
    bvh_indices.push_back(bvh_index);
    BVHNode bvh_root = construct_bvh(triangles, &bvh_index);
    std::cout << bvh_root.maxX << ", " << bvh_root.maxY << ", " << bvh_root.maxZ
              << ", " << bvh_root.minX << ", " << bvh_root.minY << ", "
              << bvh_root.minZ << std::endl;

    bvh_roots.push_back(bvh_root);
    bvh_indices.push_back(bvh_index);
    BVHNode scene_bvh_root = construct_bvh(room, &bvh_index);
    bvh_roots.push_back(scene_bvh_root);

    SDLHelper sdl;
    if (!sdl.create(width, height, "RayTrace")) {
        return 1;
    }

    Vec3 cam_pos(1.0f, 2.5f, 1.0f);
    Vec3 cam_dir(0.0f, 0.0f, -1.0f);
    Vec3 cam_up(0.0f, 1.0f, 0.0f);
    float fov = 90.0f * 3.1415926f / 180.0f;

    std::vector<Triangle> ordered_triangles;
    construct_ordered_triangles(bvh_roots[0], ordered_triangles);

    construct_ordered_triangles(bvh_roots[1], ordered_triangles);
    Kokkos::initialize(argc, argv);
    {
        Kokkos::View<gpuBVHNode *, ExecSpace> gpu_bvh_nodes =
            construct_gpu_bvh(bvh_roots, bvh_index);
        Kokkos::fence();
        Kokkos::View<Triangle *, ExecSpace> gpu_triangles =
            gpu_trigs(ordered_triangles);
        std::cout << "Total triangles: " << ordered_triangles.size()
                  << std::endl;

        Kokkos::View<int *, Kokkos::HostSpace> h_bvh_indices(
            "bvh_indices_host", bvh_indices.size());
        for (size_t i = 0; i < bvh_indices.size(); ++i)
            h_bvh_indices(i) = bvh_indices[i];

        Kokkos::View<int *, Kokkos::DefaultExecutionSpace> bvh_indices_view(
            "bvh_indices_device", bvh_indices.size());

        Kokkos::deep_copy(bvh_indices_view, h_bvh_indices);

        Kokkos::View<Lambertian *, ExecSpace> lambertians =
            make_cornell_lambertians();

        Kokkos::View<Mirror *, ExecSpace> mirrors = random_mirrors(10);
        Kokkos::View<Glass *, ExecSpace> glasses = random_glass(0);
        Kokkos::View<Emissive *, ExecSpace> emissives =
            make_cornell_emissives();

        Kokkos::View<int, ExecSpace> mirror_counter("mirror_counter");
        Kokkos::View<int, ExecSpace> lambertian_counter("lambertian_counter");
        Kokkos::View<int, ExecSpace> glass_counter("glass_counter");
        Kokkos::View<int, ExecSpace> emissive_counter("emissive_counter");
        Kokkos::View<int, ExecSpace> missed_counter("missed_counter");
        const int ray_count = width * height;

        Kokkos::View<int *, ExecSpace> mirror_indices("mirror_indices",
                                                      ray_count);
        Kokkos::View<int *, ExecSpace> lambertian_indices("lambertian_indices",
                                                          ray_count);
        Kokkos::View<int *, ExecSpace> glass_indices("glass_indices",
                                                     ray_count);
        Kokkos::View<int *, ExecSpace> emissive_indices("emissive_indices",
                                                        ray_count);
        Kokkos::View<int *, ExecSpace> missed_indices("missed_indices",
                                                      ray_count);
        Kokkos::View<Ray *, ExecSpace> rays("rays", ray_count);
        Kokkos::View<Ray *, ExecSpace> rays_new("rays_new", ray_count);

        Kokkos::View<Hit *, ExecSpace> hits("hits", width * height);
        Kokkos::View<Vec3 *, ExecSpace> colors("colors", ray_count);
        Kokkos::deep_copy(colors, Vec3(0.0f, 0.0f, 0.0f));
        Kokkos::fence();
        double total_time = 0.0;
        for (int sample = 0; sample < spp; sample++) {
            auto start = std::chrono::high_resolution_clock::now();

            uint32_t bounce_seed = 12131231 + sample * 524287;
            generate_camera_rays(rays, cam_pos, cam_dir, cam_up, fov, width,
                                 height, 1, bounce_seed);

            trace_once(width, height, rays, rays_new, hits, colors,
                       gpu_triangles, bvh_indices_view, bvh_indices.size(),
                       mirrors, glasses, lambertians, emissives, mirror_indices,
                       lambertian_indices, glass_indices, emissive_indices,
                       missed_indices, gpu_bvh_nodes, bounce_seed);
            Kokkos::fence();
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            if (sample % 10 == 1 || sample == spp - 1) {
                display(colors, width, height, sample + 1, sdl);
            }
            total_time += elapsed.count();
            std::cout << "Sample " << sample + 1 << " / " << spp << " took "
                      << elapsed.count() << " s" << std::endl;
        }
        std::cout << "Total rendering time: " << total_time << " s"
                  << std::endl;

        sdl.destroy();
    }
    Kokkos::finalize();
}
