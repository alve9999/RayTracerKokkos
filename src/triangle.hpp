#pragma once
#include "hit.hpp"
#include "material.hpp"
#include "point.hpp"
#include "ray.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <cmath>

struct Triangle {
    Vec3 v0, v1, v2;
    int materialId;
    MaterialType materialType;

    KOKKOS_INLINE_FUNCTION
    Triangle()
        : v0(Vec3(0, 0, 0)), v1(Vec3(1, 0, 0)), v2(Vec3(0, 1, 0)),
          materialType(MaterialType::None), materialId(-1) {}

    KOKKOS_INLINE_FUNCTION
    Triangle(const Vec3 &a, const Vec3 &b, const Vec3 &c, int matId,
             MaterialType matType)
        : v0(a), v1(b), v2(c), materialType(matType), materialId(matId) {}

    KOKKOS_INLINE_FUNCTION
    float intersect(const Ray &ray) const {
        const float EPSILON = 1e-8f;
        Vec3 edge1 = v1 - v0;
        Vec3 edge2 = v2 - v0;
        Vec3 h = cross(ray.direction, edge2);
        float a = dot(edge1, h);
        if (a > -EPSILON && a < EPSILON)
            return INFINITY;
        float f = 1.0f / a;
        Vec3 s = ray.origin - v0;
        float u = f * dot(s, h);
        if (u < 0.0f || u > 1.0f)
            return INFINITY;
        Vec3 q = cross(s, edge1);
        float v = f * dot(ray.direction, q);
        if (v < 0.0f || u + v > 1.0f)
            return INFINITY;
        float t = f * dot(edge2, q);
        if (t > EPSILON) {
            return t;
        } else
            return INFINITY;
    }

    KOKKOS_INLINE_FUNCTION
    Vec3 get_normal() const {
        Vec3 edge1 = v1 - v0;
        Vec3 edge2 = v2 - v0;
        return cross(edge1, edge2).normalized();
    }

    KOKKOS_INLINE_FUNCTION
    Hit get_hit_record(const Ray &ray, float t) {

        Vec3 hit_position = ray.origin + ray.direction * t;
        Vec3 normal = get_normal();
        if (dot(ray.direction, normal) > 0.0f) {
            normal = normal * -1.0f;
        }
        return Hit(t, hit_position, normal, materialId, materialType);
    }
};

struct cpuTriangle {
    Triangle *data;
    float minX, minY, minZ;
    float maxX, maxY, maxZ;
    float cX, cY, cZ;
    cpuTriangle()
        : data(nullptr), minX(0), minY(0), minZ(0), maxX(0), maxY(0), maxZ(0) {}

    cpuTriangle(Triangle *d) : data(d) {
        minX = fminf(fminf(d->v0.x, d->v1.x), d->v2.x);
        minY = fminf(fminf(d->v0.y, d->v1.y), d->v2.y);
        minZ = fminf(fminf(d->v0.z, d->v1.z), d->v2.z);
        maxX = fmaxf(fmaxf(d->v0.x, d->v1.x), d->v2.x);
        maxY = fmaxf(fmaxf(d->v0.y, d->v1.y), d->v2.y);
        maxZ = fmaxf(fmaxf(d->v0.z, d->v1.z), d->v2.z);
        cX = (minX + maxX) * 0.5f;
        cY = (minY + maxY) * 0.5f;
        cZ = (minZ + maxZ) * 0.5f;
    }
};
