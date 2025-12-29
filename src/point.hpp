#pragma once
#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <cmath>

struct Vec3 {
    float x, y, z;

    KOKKOS_INLINE_FUNCTION
    Vec3 operator-() const { return Vec3{-x, -y, -z}; }

    KOKKOS_INLINE_FUNCTION
    Vec3() : x(0), y(0), z(0) {}

    KOKKOS_INLINE_FUNCTION
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    KOKKOS_INLINE_FUNCTION
    Vec3 operator+(const Vec3 &other) const {
        return Vec3{x + other.x, y + other.y, z + other.z};
    }

    KOKKOS_INLINE_FUNCTION
    Vec3 &operator+=(const Vec3 &other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    Vec3 operator-(const Vec3 &other) const {
        return Vec3{x - other.x, y - other.y, z - other.z};
    }

    KOKKOS_INLINE_FUNCTION
    Vec3 operator*(const Vec3 &other) const {
        return Vec3{x * other.x, y * other.y, z * other.z};
    }

    KOKKOS_INLINE_FUNCTION
    Vec3 operator*(float scalar) const {
        return Vec3{x * scalar, y * scalar, z * scalar};
    }

    KOKKOS_INLINE_FUNCTION
    Vec3 operator/(float scalar) const {
        return Vec3{x / scalar, y / scalar, z / scalar};
    }

    KOKKOS_INLINE_FUNCTION
    float length() const { return std::sqrt(x * x + y * y + z * z); }

    KOKKOS_INLINE_FUNCTION float length2() const {
        return x * x + y * y + z * z;
    }
    KOKKOS_INLINE_FUNCTION Vec3 normalized() const {
        float inv_len = 1.0f / std::sqrt(length2());
        return Vec3{x * inv_len, y * inv_len, z * inv_len};
    }
    KOKKOS_INLINE_FUNCTION Vec3 &madd(const Vec3 &v, float s) {
        x += v.x * s;
        y += v.y * s;
        z += v.z * s;
        return *this;
    }
};

KOKKOS_INLINE_FUNCTION
float dot(const Vec3 &a, const Vec3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

KOKKOS_INLINE_FUNCTION
Vec3 cross(const Vec3 &a, const Vec3 &b) {
    return Vec3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x};
}
