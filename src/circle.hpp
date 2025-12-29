#pragma once
#include "hit.hpp"
#include "material.hpp"
#include "point.hpp"
#include "ray.hpp"
#include <Kokkos_Macros.hpp>
#include <cmath>

struct Circle {
    Vec3 center;
    float radius;
    int materialId;
    MaterialType materialType;

    KOKKOS_INLINE_FUNCTION
    Circle()
        : center(Vec3(0, 0, 0)), radius(1.0f), materialId(-1),
          materialType(MaterialType::None) {}

    KOKKOS_INLINE_FUNCTION
    Circle(Vec3 c, float r, int materialId, MaterialType materialType)
        : center(c), radius(r), materialId(materialId),
          materialType(materialType) {}

    KOKKOS_INLINE_FUNCTION
    Vec3 get_normal_at(Vec3 p) {
        Vec3 normal = Vec3(p.x - center.x, p.y - center.y, p.z - center.z);
        normal = normal.normalized();
        return normal;
    }

    KOKKOS_INLINE_FUNCTION
    float intersect(const Ray &ray) const {
        Vec3 oc = Vec3(ray.origin.x - center.x, ray.origin.y - center.y,
                       ray.origin.z - center.z);
        float a = dot(ray.direction, ray.direction);
        float b = 2.0f * dot(oc, ray.direction);
        float c = dot(oc, oc) - radius * radius;
        float discriminant = b * b - 4 * a * c;
        float t;

        if (discriminant < 0) {
            return INFINITY;
        } else {
            float sqrt_disc = sqrt(discriminant);
            float t0 = (-b - sqrt_disc) / (2.0f * a);
            float t1 = (-b + sqrt_disc) / (2.0f * a);

            t = (t0 < t1) ? t0 : t1;
            if (t < 0) {
                t = (t0 > t1) ? t0 : t1;
                if (t < 0)
                    return INFINITY;
            }
            return t;
        }
    }

    KOKKOS_INLINE_FUNCTION
    Hit get_hit_record(const Ray &ray, float t) {
        Vec3 hit_position = ray.origin + ray.direction * t;
        Vec3 normal = get_normal_at(hit_position);
        return Hit(t, hit_position, normal, materialId, materialType);
    }
};
