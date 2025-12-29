#pragma once
#include "circle.hpp"
#include "material.hpp"
#include "point.hpp"
#include <Kokkos_Core.hpp>

Kokkos::View<Circle *, ExecSpace> random_circles(int circle_count) {
    Kokkos::View<Circle *, ExecSpace> circles("circles", circle_count);

    unsigned int seed = 67892;
    Kokkos::Random_XorShift64_Pool<ExecSpace> rand_pool(seed);

    Kokkos::parallel_for(
        "generate_circles", Kokkos::RangePolicy<ExecSpace>(0, circle_count),
        KOKKOS_LAMBDA(int i) {
            auto gen = rand_pool.get_state();

            // 1. Define the "Floor" Sphere
            float floor_radius = 1000.0f;
            // The surface is at y = -1.0
            Vec3 floor_center(0.0f, -floor_radius - 1.0f, -15.0f);

            if (i == 0) {
                circles(i) = Circle(floor_center, floor_radius, 1,
                                    MaterialType::Lambertian);
            } else {
                float spread = 0.025f;
                float theta = (gen.drand() - 0.5f) * spread;
                float phi = (gen.drand() - 0.5f) * spread;

                float r;
                if (i % 10 == 0) {
                    r = 0.5f + gen.drand() * 2.7f; // Larger "hero" spheres
                } else {
                    r = 0.1f + gen.drand() * 0.5f; // Small "pebble" spheres
                }

                Vec3 dir(sin(theta), cos(theta) * cos(phi), sin(phi));

                Vec3 pos = floor_center + dir * (floor_radius + r);

                int materialId = i % 30;
                MaterialType materialType = (i % 3 == 2) ? MaterialType::Mirror
                                            : (i % 3 == 1)
                                                ? MaterialType::Glass
                                                : MaterialType::Lambertian;

                circles(i) = Circle(pos, r, materialId, materialType);
            }

            rand_pool.free_state(gen);
        });

    return circles;
}
Kokkos::View<Mirror *, ExecSpace> random_mirrors(int mirror_count) {
    Kokkos::View<Mirror *, ExecSpace> mirrors("mirrors", mirror_count);

    unsigned int seed = 13579;
    Kokkos::Random_XorShift64_Pool<ExecSpace> rand_pool(seed);

    Kokkos::parallel_for(
        "generate_mirrors", Kokkos::RangePolicy<ExecSpace>(0, mirror_count),
        KOKKOS_LAMBDA(int i) {
            auto gen = rand_pool.get_state();

            float r = 1.0;
            float g = 1.0;
            float b = 1.0;

            rand_pool.free_state(gen);

            mirrors(i) = Mirror(r, g, b);
        });

    return mirrors;
}

Kokkos::View<Lambertian *, ExecSpace> random_lambertians(int lambertian_count) {
    Kokkos::View<Lambertian *, ExecSpace> lambertians("lambertians",
                                                      lambertian_count);

    unsigned int seed = 24681;
    Kokkos::Random_XorShift64_Pool<ExecSpace> rand_pool(seed);

    Kokkos::parallel_for(
        "generate_lambertians",
        Kokkos::RangePolicy<ExecSpace>(0, lambertian_count),
        KOKKOS_LAMBDA(int i) {
            auto gen = rand_pool.get_state();

            float r = gen.drand();
            float g = gen.drand();
            float b = gen.drand();

            rand_pool.free_state(gen);

            lambertians(i) = Lambertian(r, g, b, 0.0f);
        });

    return lambertians;
}

Kokkos::View<Glass *, ExecSpace> random_glass(int glass_count) {
    Kokkos::View<Glass *, ExecSpace> glasses("glasses", glass_count);

    unsigned int seed = 11223;
    Kokkos::Random_XorShift64_Pool<ExecSpace> rand_pool(seed);

    Kokkos::parallel_for(
        "generate_glasses", Kokkos::RangePolicy<ExecSpace>(0, glass_count),
        KOKKOS_LAMBDA(int i) {
            auto gen = rand_pool.get_state();

            float refractive_index = 1.5f;

            rand_pool.free_state(gen);

            glasses(i) = Glass(refractive_index);
        });

    return glasses;
}
