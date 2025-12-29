#pragma once
#include "point.hpp"
#include <Kokkos_Macros.hpp>
#include <Kokkos_Random.hpp>
#include <decl/Kokkos_Declare_CUDA.hpp>
using ExecSpace = Kokkos::Cuda;

enum class MaterialType : int { None, Lambertian, Mirror, Glass, Emissive };

struct Mirror {
    float reflectivity_r; // Reflectivity coefficient
    float reflectivity_g;
    float reflectivity_b;

    KOKKOS_INLINE_FUNCTION
    Mirror()
        : reflectivity_r(1.0f), reflectivity_g(1.0f), reflectivity_b(1.0f) {}

    KOKKOS_INLINE_FUNCTION
    Mirror(float r, float g, float b)
        : reflectivity_r(r), reflectivity_g(g), reflectivity_b(b) {}

    KOKKOS_INLINE_FUNCTION
    Vec3 reflect(const Vec3 &incident, const Vec3 &normal) const {
        return (incident - normal * (2.0f * dot(incident, normal)))
            .normalized();
    }

    KOKKOS_INLINE_FUNCTION
    Vec3 reflectivity_color(const Vec3 &base_color) const {
        return Vec3(base_color.x * reflectivity_r,
                    base_color.y * reflectivity_g,
                    base_color.z * reflectivity_b);
    }
};

struct Lambertian {
    float albedo_r; // Albedo coefficient
    float albedo_g;
    float albedo_b;
    float specularity = 0.0f; // Specularity coefficient (0.0 = diffuse, 1.0 =
                              // perfect mirror)

    KOKKOS_INLINE_FUNCTION
    Lambertian()
        : albedo_r(0.8f), albedo_g(0.8f), albedo_b(0.8f), specularity(0.0f) {}

    KOKKOS_INLINE_FUNCTION
    Lambertian(float r, float g, float b, float spec)
        : albedo_r(r), albedo_g(g), albedo_b(b), specularity(spec) {}

    template <typename RNG>
    KOKKOS_INLINE_FUNCTION Vec3 random_diffuse_direction(const Vec3 &normal,
                                                         RNG &rand) {
        float u1 = rand();
        float u2 = rand();

        float r = sqrt(u1);
        float theta = 2.0f * 3.1415926f * u2;

        float x = r * cos(theta);
        float y = r * sin(theta);
        float z = sqrt(1.0f - u1);

        Vec3 tangent, bitangent;
        if (fabs(normal.x) > fabs(normal.y)) {
            tangent = Vec3(normal.z, 0, -normal.x).normalized();
        } else {
            tangent = Vec3(0, -normal.z, normal.y).normalized();
        }
        bitangent = cross(normal, tangent);

        Vec3 diffuse_dir = tangent * x + bitangent * y + normal * z;
        return diffuse_dir.normalized();
    }

    KOKKOS_INLINE_FUNCTION
    Vec3 diffuse_color(const Vec3 &base_color) const {
        return Vec3(base_color.x * albedo_r, base_color.y * albedo_g,
                    base_color.z * albedo_b);
    }
};

struct Glass {
    float refractive_index; // Refractive index

    KOKKOS_INLINE_FUNCTION
    Glass() : refractive_index(1.5f) {}

    KOKKOS_INLINE_FUNCTION
    Glass(float ri) : refractive_index(ri) {}

    KOKKOS_INLINE_FUNCTION
    Vec3 refract(const Vec3 &incident, const Vec3 &normal,
                 bool entering) const {
        float eta = entering ? (1.0f / refractive_index) : refractive_index;
        float cosi = -dot(normal, incident);
        float sint2 = eta * eta * (1.0f - cosi * cosi);
        if (sint2 > 1.0f) {
            return Vec3(0, 0, 0);
        }
        float cost = sqrt(1.0f - sint2);
        return (incident * eta + normal * (eta * cosi - cost)).normalized();
    }

    KOKKOS_INLINE_FUNCTION
    Vec3 reflect(const Vec3 &incident, const Vec3 &normal) const {
        return (incident - normal * (2.0f * dot(incident, normal)))
            .normalized();
    }

    KOKKOS_INLINE_FUNCTION
    float schlick_approximation(float cos_i, float eta_i, float eta_t) const {
        float r0 = (eta_i - eta_t) / (eta_i + eta_t);
        r0 = r0 * r0;

        float cos_x = cos_i;

        if (eta_i > eta_t) {
            float n = eta_i / eta_t;
            float sin2_t = n * n * (1.0f - cos_i * cos_i);

            if (sin2_t > 1.0f)
                return 1.0f;

            cos_x = sqrt(1.0f - sin2_t);
        }

        float x = 1.0f - cos_x;
        return r0 + (1.0f - r0) * (x * x * x * x * x);
    }
};

struct Emissive {
    Vec3 emit;
    KOKKOS_INLINE_FUNCTION
    Emissive() : emit() {}

    KOKKOS_INLINE_FUNCTION
    Emissive(float r, float g, float b) : emit(r, g, b) {}
};
