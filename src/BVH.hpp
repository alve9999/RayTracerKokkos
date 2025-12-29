#pragma once
#include "point.hpp"
#include "triangle.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <assimp/mesh.h>
#include <iostream>
#include <vector>

using ExecSpace = Kokkos::Cuda;

struct BVHAccum {
    float minX = INFINITY, minY = INFINITY, minZ = INFINITY;
    float maxX = -INFINITY, maxY = -INFINITY, maxZ = -INFINITY;
    int count = 0;

    void grow(const cpuTriangle &t) {
        minX = std::min(minX, t.minX);
        minY = std::min(minY, t.minY);
        minZ = std::min(minZ, t.minZ);
        maxX = std::max(maxX, t.maxX);
        maxY = std::max(maxY, t.maxY);
        maxZ = std::max(maxZ, t.maxZ);
        ++count;
    }

    float surface_area() const {
        float dx = maxX - minX;
        float dy = maxY - minY;
        float dz = maxZ - minZ;
        return 2.0f * (dx * dy + dy * dz + dz * dx);
    }
};

struct BVHNode {
    BVHNode *left;
    BVHNode *right;
    float minX, minY, minZ;
    float maxX, maxY, maxZ;
    int depth = 0;
    int start, end;
    int index = 0;
    int size;
    float cost;
    std::vector<cpuTriangle> triangles;

    float surface_area() const {
        float dx = maxX - minX;
        float dy = maxY - minY;
        float dz = maxZ - minZ;
        return 2.0f * (dx * dy + dy * dz + dz * dx);
    }

    BVHNode()
        : left(nullptr), right(nullptr), minX(INFINITY), minY(INFINITY),
          minZ(INFINITY), maxX(-INFINITY), maxY(-INFINITY), maxZ(-INFINITY) {}

    void grow_to_include(cpuTriangle &tri) {
        // Update bounding box to include triangle

        minX = std::min(minX, tri.minX);
        minY = std::min(minY, tri.minY);
        minZ = std::min(minZ, tri.minZ);
        maxX = std::max(maxX, tri.maxX);
        maxY = std::max(maxY, tri.maxY);
        maxZ = std::max(maxZ, tri.maxZ);
        triangles.push_back(tri);
    }

    void clone_from(const BVHAccum &other) {
        minX = other.minX;
        minY = other.minY;
        minZ = other.minZ;
        maxX = other.maxX;
        maxY = other.maxY;
        maxZ = other.maxZ;
    }

    void add_to(cpuTriangle &tri) { triangles.push_back(tri); }

    void split() {
        if (depth >= 32 || triangles.size() <= 2) {
            return;
        }

        float best_cost = cost;
        int best_axis = -1;
        float best_split = 0.0f;
        float best_left = 0.0f;
        float best_right = 0.0f;
        BVHAccum best_left_accum, best_right_accum;

        for (int axis = 0; axis < 3; axis++) {
            std::vector<float> centers;
            size_t n = triangles.size();
            for (size_t i = 0; i < 5; ++i) {
                size_t idx = i * n / 5;
                if (idx >= n)
                    idx = n - 1;
                auto &tri = triangles[idx];
                float c = 0.0f;
                if (axis == 0)
                    c = 0.5f * (tri.minX + tri.maxX);
                else if (axis == 1)
                    c = 0.5f * (tri.minY + tri.maxY);
                else
                    c = 0.5f * (tri.minZ + tri.maxZ);
                centers.push_back(c);
            }

            for (float split : centers) {
                BVHAccum leftA, rightA;

                for (auto &tri : triangles) {
                    float c = axis == 0   ? 0.5f * (tri.minX + tri.maxX)
                              : axis == 1 ? 0.5f * (tri.minY + tri.maxY)
                                          : 0.5f * (tri.minZ + tri.maxZ);

                    if (c < split)
                        leftA.grow(tri);
                    else
                        rightA.grow(tri);
                }

                if (leftA.count == 0 || rightA.count == 0)
                    continue;

                float cost = 1.0f + leftA.surface_area() * leftA.count +
                             rightA.surface_area() * rightA.count;

                if (cost < best_cost) {
                    best_left = leftA.count * leftA.surface_area();
                    best_right = rightA.count * rightA.surface_area();
                    best_cost = cost;
                    best_axis = axis;
                    best_split = split;
                    best_left_accum = leftA;
                    best_right_accum = rightA;
                }
            }
        }

        if (!best_left || !best_right) {
            return;
        }
        if (best_cost >= cost) {
            return;
        }

        left = new BVHNode();
        right = new BVHNode();

        left->clone_from(best_left_accum);
        right->clone_from(best_right_accum);
        for (auto &tri : triangles) {
            float c = best_axis == 0   ? 0.5f * (tri.minX + tri.maxX)
                      : best_axis == 1 ? 0.5f * (tri.minY + tri.maxY)
                                       : 0.5f * (tri.minZ + tri.maxZ);

            if (c < best_split)
                left->add_to(tri);
            else
                right->add_to(tri);
        }

        triangles.clear();

        left->depth = depth + 1;
        right->depth = depth + 1;
        left->cost = best_left;
        right->cost = best_right;

        left->split();
        right->split();
    }
};

struct gpuBVHNode {
    float minX, minY, minZ;
    float maxX, maxY, maxZ;
    int index;
    int triangleCount;

    KOKKOS_INLINE_FUNCTION
    gpuBVHNode()
        : minX(INFINITY), minY(INFINITY), minZ(INFINITY), maxX(-INFINITY),
          maxY(-INFINITY), maxZ(-INFINITY), index(-1), triangleCount(0) {}

    KOKKOS_INLINE_FUNCTION
    bool is_leaf() const { return triangleCount > 0; }

    KOKKOS_INLINE_FUNCTION
    bool intersect(const Ray &ray, float &t_min, float &t_max) const {
        float tx0 = (minX - ray.origin.x) * ray.inv_dir.x;
        float tx1 = (maxX - ray.origin.x) * ray.inv_dir.x;
        float tmin = fminf(tx0, tx1);
        float tmax = fmaxf(tx0, tx1);

        float ty0 = (minY - ray.origin.y) * ray.inv_dir.y;
        float ty1 = (maxY - ray.origin.y) * ray.inv_dir.y;
        tmin = fmaxf(tmin, fminf(ty0, ty1));
        tmax = fminf(tmax, fmaxf(ty0, ty1));

        float tz0 = (minZ - ray.origin.z) * ray.inv_dir.z;
        float tz1 = (maxZ - ray.origin.z) * ray.inv_dir.z;
        tmin = fmaxf(tmin, fminf(tz0, tz1));
        tmax = fminf(tmax, fmaxf(tz0, tz1));

        return tmax >= fmaxf(tmin, t_min) && tmin <= t_max;
    }
};

void number_bvh(BVHNode &node, int *current_index) {
    if (node.left && node.right) {
        (*current_index)++;
        node.left->index = *current_index;
        (*current_index)++;
        node.right->index = *current_index;
    } else {
        return;
    }
    if (node.left) {
        number_bvh(*node.left, current_index);
    }
    if (node.right) {
        number_bvh(*node.right, current_index);
    }
}

std::vector<cpuTriangle> create_scene_geometry() {
    std::vector<cpuTriangle> scene_triangles;

    Vec3 pmin(-0.5f, 1.0f, -3.77013f);
    Vec3 pmax(2.5, 7.0, 2.0f);

    scene_triangles.push_back(cpuTriangle(
        new Triangle(Vec3(pmin.x, pmin.y, pmin.z), Vec3(pmax.x, pmin.y, pmin.z),
                     Vec3(pmax.x, pmin.y, pmax.z), 0, MaterialType::Mirror)));
    scene_triangles.push_back(cpuTriangle(
        new Triangle(Vec3(pmin.x, pmin.y, pmin.z), Vec3(pmax.x, pmin.y, pmax.z),
                     Vec3(pmin.x, pmin.y, pmax.z), 0, MaterialType::Mirror)));

    // --- Walls ---
    // Front wall (along z = pmax.z)
    scene_triangles.push_back(cpuTriangle(new Triangle(
        Vec3(pmin.x, pmin.y, pmax.z), Vec3(pmax.x, pmin.y, pmax.z),
        Vec3(pmax.x, pmax.y, pmax.z), 0, MaterialType::Lambertian)));
    scene_triangles.push_back(cpuTriangle(new Triangle(
        Vec3(pmin.x, pmin.y, pmax.z), Vec3(pmax.x, pmax.y, pmax.z),
        Vec3(pmin.x, pmax.y, pmax.z), 0, MaterialType::Lambertian)));

    // Back wall (z = pmin.z)
    scene_triangles.push_back(cpuTriangle(new Triangle(
        Vec3(pmin.x, pmin.y, pmin.z), Vec3(pmax.x, pmax.y, pmin.z),
        Vec3(pmax.x, pmin.y, pmin.z), 0, MaterialType::Lambertian)));
    scene_triangles.push_back(cpuTriangle(new Triangle(
        Vec3(pmin.x, pmin.y, pmin.z), Vec3(pmin.x, pmax.y, pmin.z),
        Vec3(pmax.x, pmax.y, pmin.z), 0, MaterialType::Lambertian)));

    // Left wall (x = pmin.x)
    scene_triangles.push_back(cpuTriangle(new Triangle(
        Vec3(pmin.x, pmin.y, pmin.z), Vec3(pmin.x, pmin.y, pmax.z),
        Vec3(pmin.x, pmax.y, pmax.z), 0, MaterialType::Lambertian)));
    scene_triangles.push_back(cpuTriangle(new Triangle(
        Vec3(pmin.x, pmin.y, pmin.z), Vec3(pmin.x, pmax.y, pmax.z),
        Vec3(pmin.x, pmax.y, pmin.z), 0, MaterialType::Lambertian)));

    // Right wall (x = pmax.x)
    scene_triangles.push_back(cpuTriangle(new Triangle(
        Vec3(pmax.x, pmin.y, pmin.z), Vec3(pmax.x, pmax.y, pmax.z),
        Vec3(pmax.x, pmin.y, pmax.z), 0, MaterialType::Lambertian)));
    scene_triangles.push_back(cpuTriangle(new Triangle(
        Vec3(pmax.x, pmin.y, pmin.z), Vec3(pmax.x, pmax.y, pmin.z),
        Vec3(pmax.x, pmax.y, pmax.z), 0, MaterialType::Lambertian)));

    // --- Roof ---
    /*scene_triangles.push_back(cpuTriangle(new Triangle(
        Vec3(pmin.x, pmax.y, pmin.z + 0.0), Vec3(pmax.x, pmax.y, pmin.z + 0.0),
        Vec3(pmax.x, pmax.y, pmax.z), 0, MaterialType::Lambertian)));
    scene_triangles.push_back(cpuTriangle(new Triangle(
        Vec3(pmin.x, pmax.y, pmin.z + 0.0), Vec3(pmax.x, pmax.y, pmax.z),
        Vec3(pmin.x, pmax.y, pmax.z), 0, MaterialType::Lambertian)));*/
    // --- Lamp ---
    /*Vec3 lamp_center(0.f, pmax.y - 0.2f, -2.f);
    float lamp_size = 1.f;
    Vec3 lamp_min = lamp_center + Vec3(-lamp_size, 0.f, -lamp_size);
    Vec3 lamp_max = lamp_center + Vec3(lamp_size, 0.f, lamp_size);

    scene_triangles.push_back(
        cpuTriangle(new Triangle(Vec3(lamp_min.x, lamp_center.y, lamp_min.z),
                                 Vec3(lamp_max.x, lamp_center.y, lamp_min.z),
                                 Vec3(lamp_max.x, lamp_center.y, lamp_max.z), 0,
                                 MaterialType::Emissive)));
    scene_triangles.push_back(
        cpuTriangle(new Triangle(Vec3(lamp_min.x, lamp_center.y, lamp_min.z),
                                 Vec3(lamp_max.x, lamp_center.y, lamp_max.z),
                                 Vec3(lamp_min.x, lamp_center.y, lamp_max.z), 0,
                                 MaterialType::Emissive)));*/

    return scene_triangles;
}

void gather_triangles(aiMesh *mesh, std::vector<cpuTriangle> &triangles) {
    for (unsigned f = 0; f < mesh->mNumFaces; ++f) {
        const aiFace &face = mesh->mFaces[f];

        if (face.mNumIndices != 3)
            continue;
        const aiVector3D &a = mesh->mVertices[face.mIndices[0]];
        const aiVector3D &b = mesh->mVertices[face.mIndices[1]];
        const aiVector3D &c = mesh->mVertices[face.mIndices[2]];
        Triangle *tri;
        tri = new Triangle(Vec3(a.x, a.y, a.z), Vec3(b.x, b.y, b.z),
                           Vec3(c.x, c.y, c.z), 1, MaterialType::Lambertian);
        cpuTriangle cpuTri(tri);
        triangles.push_back(cpuTri);
    }
}

BVHNode construct_bvh(std::vector<cpuTriangle> &trigs, int *index) {
    BVHNode node;

    for (auto &tri : trigs) {
        node.grow_to_include(tri);
    }

    float sa = (node.maxX - node.minX) * (node.maxY - node.minY) +
               (node.maxY - node.minY) * (node.maxZ - node.minZ) +
               (node.maxZ - node.minZ) * (node.maxX - node.minX);
    node.cost = sa * node.triangles.size();

    node.split();
    node.index = *index;
    number_bvh(node, index);
    return node;
}

template <typename ViewType>
void construct_gpu_bvh_rec(BVHNode &node, ViewType &gpu_nodes) {
    gpuBVHNode gpu_node;
    gpu_node.minX = node.minX;
    gpu_node.minY = node.minY;
    gpu_node.minZ = node.minZ;
    gpu_node.maxX = node.maxX;
    gpu_node.maxY = node.maxY;
    gpu_node.maxZ = node.maxZ;
    gpu_node.triangleCount = node.triangles.size();
    if (!node.left && !node.right) {
        gpu_node.index = node.start;
        gpu_nodes(node.index) = gpu_node;
        return;
    } else {
        gpu_node.index = node.left->index;
        gpu_nodes(node.index) = gpu_node;
    }

    if (node.left) {
        construct_gpu_bvh_rec<ViewType>(*node.left, gpu_nodes);
    }
    if (node.right) {
        construct_gpu_bvh_rec<ViewType>(*node.right, gpu_nodes);
    }
}

Kokkos::View<gpuBVHNode *, ExecSpace>
construct_gpu_bvh(std::vector<BVHNode> &bvh_roots, int size) {
    Kokkos::View<gpuBVHNode *, ExecSpace> gpu_nodes("gpu_bvh_nodes", size + 1);
    auto h = Kokkos::create_mirror_view(gpu_nodes);
    gpuBVHNode gpu_node;
    for (auto &node : bvh_roots) {
        construct_gpu_bvh_rec<decltype(h)>(node, h);
    }
    Kokkos::deep_copy(gpu_nodes, h);
    return gpu_nodes;
}

void construct_ordered_triangles(BVHNode &node,
                                 std::vector<Triangle> &ordered_triangles) {
    if (node.left == nullptr && node.right == nullptr) {
        node.start = ordered_triangles.size();
        for (auto &cpuTri : node.triangles) {
            ordered_triangles.push_back(*(cpuTri.data));
        }
        node.end = ordered_triangles.size();
        return;
    }
    if (node.left) {
        construct_ordered_triangles(*node.left, ordered_triangles);
    }
    if (node.right) {
        construct_ordered_triangles(*node.right, ordered_triangles);
    }
}

KOKKOS_INLINE_FUNCTION
Hit traverse_bvh(const Ray &ray,
                 const Kokkos::View<gpuBVHNode *, ExecSpace> &bvh_nodes,
                 const Kokkos::View<Triangle *, ExecSpace> &triangles,
                 int root_index) {
    Hit best_hit;
    best_hit.t = INFINITY;
    best_hit.materialType = MaterialType::None;

    int stack[32];
    int stack_ptr = 0;
    stack[stack_ptr++] = root_index;

    float t_min = 0.0f;
    float t_max = INFINITY;
    if (!bvh_nodes(root_index).intersect(ray, t_min, t_max)) {
        return best_hit;
    }

    while (stack_ptr > 0) {
        int node_idx = stack[--stack_ptr];
        const gpuBVHNode &node = bvh_nodes(node_idx);

        if (t_min > best_hit.t) {
            continue;
        }

        if (node.is_leaf()) {
            int tri_start = node.index;
            int tri_end = tri_start + node.triangleCount;

            for (int i = tri_start; i < tri_end; i++) {

                float t = triangles(i).intersect(ray);
                if (t > 0.0f && t < best_hit.t) {

                    best_hit = triangles(i).get_hit_record(ray, t);
                }
            }
        } else {
            int left_idx = node.index;
            int right_idx = node.index + 1;

            float t_min_left = 0.0f, t_max_left = best_hit.t;
            float t_min_right = 0.0f, t_max_right = best_hit.t;

            bool hit_left =
                bvh_nodes(left_idx).intersect(ray, t_min_left, t_max_left);
            bool hit_right =
                bvh_nodes(right_idx).intersect(ray, t_min_right, t_max_right);

            if (hit_left && hit_right) {
                if (t_min_left < t_min_right) {
                    stack[stack_ptr++] = right_idx;
                    stack[stack_ptr++] = left_idx;
                } else {
                    stack[stack_ptr++] = left_idx;
                    stack[stack_ptr++] = right_idx;
                }
            } else if (hit_left) {
                stack[stack_ptr++] = left_idx;
            } else if (hit_right) {
                stack[stack_ptr++] = right_idx;
            }
        }
    }

    return best_hit;
}
