#pragma once
#include "material.hpp"
#include "point.hpp"
#include <Kokkos_Macros.hpp>
#include <cmath>

struct Hit {
    float t;
    Vec3 position;
    Vec3 normal;
    int materialId;
    MaterialType materialType;

    KOKKOS_INLINE_FUNCTION
    Hit()
        : t(INFINITY), position(Vec3()), normal(Vec3()), materialId(-1),
          materialType(MaterialType::None) {}

    KOKKOS_INLINE_FUNCTION
    Hit(float t, const Vec3 &position, const Vec3 &normal, int materialId,
        MaterialType materialType)
        : t(t), position(position), normal(normal), materialId(materialId),
          materialType(materialType) {}
};
