#pragma once
#include "point.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>

struct Ray {
    int pixelIndex;
    Vec3 origin;
    Vec3 direction;
    Vec3 inv_dir;
    Vec3 color;

    KOKKOS_INLINE_FUNCTION
    Ray(int pixelIndex, const Vec3 &o, const Vec3 &d, const Vec3 &c)
        : pixelIndex(pixelIndex), origin(o), direction(d.normalized()),
          color(c) {
        inv_dir =
            Vec3(1.0f / direction.x, 1.0f / direction.y, 1.0f / direction.z);
    }

    KOKKOS_INLINE_FUNCTION
    Vec3 at(float t) const { return origin + direction * t; }

    KOKKOS_INLINE_FUNCTION
    Ray() : pixelIndex(0), origin(), direction(0, 0, 1), color(0.0, 0.0, 0.0) {}
};
