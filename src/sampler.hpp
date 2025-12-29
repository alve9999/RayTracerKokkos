#include "BVH.hpp"
#include "circle.hpp"
#include "material.hpp"
#include "point.hpp"
#include "ray.hpp"
#include "triangle.hpp"
#include <Kokkos_Core.hpp>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <decl/Kokkos_Declare_CUDA.hpp>
#include <decl/Kokkos_Declare_OPENMP.hpp>
#include <iostream>
#include <vector>

using ExecSpace = Kokkos::Cuda;

void generate_camera_rays(Kokkos::View<Ray *, ExecSpace> rays, Vec3 cam_pos,
                          Vec3 cam_dir, Vec3 cam_up, float fov, int width,
                          int height, int spp, int seed) {
    const float aspect = float(width) / float(height);
    const float scale = tanf(fov * 0.5f);

    Vec3 right = cross(cam_dir, cam_up).normalized();
    ;
    Kokkos::parallel_for(
        "camera_rays", Kokkos::RangePolicy<ExecSpace>(0, width * height * spp),
        KOKKOS_LAMBDA(int idx) {
            int pixel = idx / spp;
            int sample = idx % spp;
            int i = pixel % width;
            int j = pixel / width;

            uint64_t state = seed + idx * 0x9E3779B97F4A7C15ULL;
            auto rand = [&]() {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                return float(state & 0xFFFFFFFF) / float(0xFFFFFFFF);
            };
            float u = (i + rand()) / float(width);
            float v = (j + rand()) / float(height);

            float px = (2.0f * u - 1.0f) * aspect * scale;
            float py = (1.0f - 2.0f * v) * scale;

            Vec3 dir = (cam_dir + right * px + cam_up * py).normalized();
            rays(idx).origin = cam_pos;
            rays(idx).direction = dir;
            rays(idx).inv_dir = Vec3(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);
            rays(idx).pixelIndex = pixel;
            rays(idx).color = Vec3(1.0f, 1.0f, 1.0f);
        });
}

Kokkos::View<Triangle *, ExecSpace>
gpu_trigs(std::vector<Triangle> &triangles) {
    int tri_count = triangles.size();

    Kokkos::View<Triangle *, ExecSpace> tris("trigs", tri_count);
    auto h = Kokkos::create_mirror_view(tris);

    int t = 0;
    for (const Triangle &tri : triangles) {
        h(t++) = tri;
    }
    Kokkos::deep_copy(tris, h);
    return tris;
}

Kokkos::View<Lambertian *, ExecSpace> make_cornell_lambertians() {
    Kokkos::View<Lambertian *, ExecSpace> mats("lambertians", 3);
    auto h = Kokkos::create_mirror_view(mats);

    h(0) = Lambertian(0.73f, 0.73f, 0.73f, 0.0f);
    h(1) = Lambertian(0.05f, 0.05f, 0.05f, 0.5f);
    h(2) = Lambertian(0.12f, 0.45f, 0.15f, 0.0f);

    Kokkos::deep_copy(mats, h);
    return mats;
}

Kokkos::View<Emissive *, ExecSpace> make_cornell_emissives() {
    Kokkos::View<Emissive *, ExecSpace> mats("emissives", 1);
    auto h = Kokkos::create_mirror_view(mats);

    h(0) = Emissive(5.0f, 5.0f, 5.0f);

    Kokkos::deep_copy(mats, h);
    return mats;
}

void intersect_triangles(Kokkos::View<Ray *, ExecSpace> rays,
                         Kokkos::View<Hit *, ExecSpace> hits,
                         Kokkos::View<gpuBVHNode *, ExecSpace> bvh_nodes,
                         Kokkos::View<int *, ExecSpace> bvh_roots,
                         int root_count,
                         Kokkos::View<Triangle *, ExecSpace> triangles, int n) {
    Kokkos::parallel_for(
        "bvh_intersect", Kokkos::RangePolicy<ExecSpace>(0, n),
        KOKKOS_LAMBDA(int i) {
            if (rays(i).pixelIndex < 0) {
                hits(i) = Hit();
                return;
            }
            Hit best_hit = Hit();
            for (int r = 0; r < root_count; r++) {
                int root_index = bvh_roots(r);
                Hit hit =
                    traverse_bvh(rays(i), bvh_nodes, triangles, root_index);
                if (hit.t < best_hit.t) {
                    best_hit = hit;
                }
            }
            hits(i) = best_hit;
        });
}

Kokkos::View<Hit *, ExecSpace>
intersect_circles(Kokkos::View<Ray *, ExecSpace> rays,
                  Kokkos::View<Circle *, ExecSpace> circle,
                  Kokkos::View<Hit *, ExecSpace> hits, int n,
                  int circle_count) {

    Kokkos::parallel_for(
        "circle_intersect", Kokkos::RangePolicy<ExecSpace>(0, n),
        KOKKOS_LAMBDA(int i) {
            if (rays(i).pixelIndex < 0) {
                hits(i) = Hit();
                return;
            }
            float best_t = INFINITY;
            int c = -1;
            for (int j = 0; j < circle_count; j++) {
                float t = circle(j).intersect(rays(i));
                if (t < best_t) {
                    c = j;
                    best_t = t;
                }
            }
            if (c == -1) {
                hits(i) = Hit();
                return;
            }
            hits(i) = circle(c).get_hit_record(rays(i), best_t);
        });

    return hits;
}

void caracterize_hits(Kokkos::View<Hit *, ExecSpace> hits,
                      Kokkos::View<int *, ExecSpace> mirror_indices,
                      Kokkos::View<int *, ExecSpace> lambertian_indices,
                      Kokkos::View<int *, ExecSpace> glass_indices,
                      Kokkos::View<int *, ExecSpace> emissive_indices,
                      Kokkos::View<int *, ExecSpace> missed_indices,
                      Kokkos::View<int, ExecSpace> mirror_count,
                      Kokkos::View<int, ExecSpace> lambertian_count,
                      Kokkos::View<int, ExecSpace> glass_count,
                      Kokkos::View<int, ExecSpace> emissive_count,
                      Kokkos::View<int, ExecSpace> missed_count, int n) {
    Kokkos::parallel_for(
        "caracterize", Kokkos::RangePolicy<ExecSpace>(0, n),
        KOKKOS_LAMBDA(int i) {
            Hit &hit = hits(i);
            if (hit.materialType == MaterialType::Mirror) {
                int idx = Kokkos::atomic_fetch_add(&mirror_count(), 1);
                mirror_indices(idx) = i;
            } else if (hit.materialType == MaterialType::Lambertian) {
                int idx = Kokkos::atomic_fetch_add(&lambertian_count(), 1);
                lambertian_indices(idx) = i;
            } else if (hit.materialType == MaterialType::Glass) {
                int idx = Kokkos::atomic_fetch_add(&glass_count(), 1);
                glass_indices(idx) = i;
            } else if (hit.materialType == MaterialType::Emissive) {
                int idx = Kokkos::atomic_fetch_add(&emissive_count(), 1);
                int idx_m = Kokkos::atomic_fetch_add(&missed_count(), 1);
                missed_indices(idx) = i;
                emissive_indices(idx) = i;
            } else {
                int idx = Kokkos::atomic_fetch_add(&missed_count(), 1);
                missed_indices(idx) = i;
            }
        });
}

void handle_missed_rays(Kokkos::View<Ray *, ExecSpace> new_rays,
                        Kokkos::View<Ray *, ExecSpace> old_rays,
                        Kokkos::View<Hit *, ExecSpace> hits,
                        Kokkos::View<int *, ExecSpace> missed_indices,
                        Kokkos::View<Vec3 *, ExecSpace> colors, int n) {
    Kokkos::parallel_for(
        "handle_missed_rays", Kokkos::RangePolicy<ExecSpace>(0, n),
        KOKKOS_LAMBDA(int i) {
            int hit_index = missed_indices(i);
            int pixel = old_rays(hit_index).pixelIndex;
            if (pixel < 0)
                return;

            Vec3 d = old_rays(hit_index).direction.normalized();
            float t = 0.5f * (d.y + 1.0f);
            Vec3 sky_color = Vec3(0.5f, 0.7f, 1.0f);
            Vec3 horizon_color = Vec3(0.7f, 0.8f, 0.9f);
            Vec3 sky = horizon_color * (1.0f - t) + sky_color * t;

            Vec3 sun_dir = Vec3(0.3f, 0.9f, 0.2f).normalized();
            float cos_theta = fmaxf(dot(d, sun_dir), 0.0f);
            float sun = pow(cos_theta, 500.0f);
            Vec3 sun_color = Vec3(20.0f, 18.0f, 14.0f) * sun;

            Vec3 L = (sky + sun_color) * old_rays(hit_index).color;
            // L = Vec3(0.5f, 0.7f, 1.0f) * 3.0f;

            Kokkos::atomic_add(&colors(pixel).x, L.x);
            Kokkos::atomic_add(&colors(pixel).y, L.y);
            Kokkos::atomic_add(&colors(pixel).z, L.z);

            new_rays(hit_index).pixelIndex = -1;
            old_rays(hit_index).pixelIndex = -1;
        });
}

void handle_emissive_hits(Kokkos::View<Ray *, ExecSpace> new_rays,
                          Kokkos::View<Ray *, ExecSpace> old_rays,
                          Kokkos::View<Hit *, ExecSpace> hits,
                          Kokkos::View<int *, ExecSpace> emissive_indices,
                          Kokkos::View<Emissive *, ExecSpace> emissives,
                          Kokkos::View<Vec3 *, ExecSpace> colors, int n) {
    Kokkos::parallel_for(
        "handle_emissive_hits", Kokkos::RangePolicy<ExecSpace>(0, n),
        KOKKOS_LAMBDA(int i) {
            int hit_index = emissive_indices(i);
            int pixel = old_rays(hit_index).pixelIndex;
            if (pixel < 0)
                return;
            Emissive mat = emissives(hits(hit_index).materialId);

            Vec3 L = mat.emit * old_rays(hit_index).color;

            Kokkos::atomic_add(&colors(pixel).x, L.x);
            Kokkos::atomic_add(&colors(pixel).y, L.y);
            Kokkos::atomic_add(&colors(pixel).z, L.z);

            new_rays(hit_index).pixelIndex = -1;
        });
}

void handle_mirror_rays(Kokkos::View<Ray *, ExecSpace> new_rays,
                        Kokkos::View<Ray *, ExecSpace> old_rays,
                        Kokkos::View<Hit *, ExecSpace> hits,
                        Kokkos::View<int *, ExecSpace> mirror_indices,
                        Kokkos::View<Mirror *, ExecSpace> mirrors, int n) {
    Kokkos::parallel_for(
        "handle_mirror_rays", Kokkos::RangePolicy<ExecSpace>(0, n),
        KOKKOS_LAMBDA(int i) {
            int hit_index = mirror_indices(i);
            Hit &hit = hits(hit_index);
            Mirror mat = mirrors(hit.materialId);
            Vec3 reflected_dir =
                mat.reflect(old_rays(hit_index).direction, hit.normal);
            new_rays(hit_index).origin = hit.position + hit.normal * 0.001f;
            new_rays(hit_index).direction = reflected_dir;
            new_rays(hit_index).inv_dir =
                Vec3(1.0f / reflected_dir.x, 1.0f / reflected_dir.y,
                     1.0f / reflected_dir.z);
            new_rays(hit_index).color =
                mat.reflectivity_color(old_rays(hit_index).color);
            new_rays(hit_index).pixelIndex = old_rays(hit_index).pixelIndex;
        });
}

void handle_glass_rays(Kokkos::View<Ray *, ExecSpace> new_rays,
                       Kokkos::View<Ray *, ExecSpace> old_rays,
                       Kokkos::View<Hit *, ExecSpace> hits,
                       Kokkos::View<int *, ExecSpace> glass_indices,
                       Kokkos::View<Glass *, ExecSpace> glasses,
                       int global_seed, int n) {
    Kokkos::parallel_for(
        "glass_rays", Kokkos::RangePolicy<ExecSpace>(0, n),
        KOKKOS_LAMBDA(int i) {
            int hit_index = glass_indices(i);
            Hit &hit = hits(hit_index);
            Glass mat = glasses(hit.materialId);

            uint64_t state = global_seed + i * 0x9E3779B97F4A7C15ULL;
            auto rand = [&]() {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                return float(state & 0xFFFFFFFF) / float(0xFFFFFFFF);
            };
            float select_rand = rand();

            Vec3 in_dir = old_rays(hit_index).direction;
            Vec3 outward_normal = hit.normal;

            float dot_in_n = dot(in_dir, outward_normal);
            bool entering = dot_in_n < 0.0f;

            Vec3 m_normal = entering ? outward_normal : -outward_normal;

            float eta_i = entering ? 1.0f : mat.refractive_index;
            float eta_t = entering ? mat.refractive_index : 1.0f;

            float cosi = entering ? -dot_in_n : dot(in_dir, outward_normal);
            float fresnel = mat.schlick_approximation(cosi, eta_i, eta_t);

            Vec3 refr = mat.refract(in_dir, m_normal, entering);
            bool total_internal_reflection = (refr.length() == 0.0f);

            if (total_internal_reflection || select_rand < fresnel) {
                Vec3 refl = mat.reflect(in_dir, m_normal);
                new_rays(hit_index).origin = hit.position + m_normal * 0.0001f;
                new_rays(hit_index).direction = refl;
                new_rays(hit_index).inv_dir =
                    Vec3(1.0f / refl.x, 1.0f / refl.y, 1.0f / refl.z);
                new_rays(hit_index).color = old_rays(hit_index).color;
            } else {
                new_rays(hit_index).origin = hit.position - m_normal * 0.0001f;
                new_rays(hit_index).direction = refr;
                new_rays(hit_index).inv_dir =
                    Vec3(1.0f / refr.x, 1.0f / refr.y, 1.0f / refr.z);
                new_rays(hit_index).color = old_rays(hit_index).color;
            }

            new_rays(hit_index).pixelIndex = old_rays(hit_index).pixelIndex;
        });
}

void spawn_lambertian_samples(int lambertian_count, int missed_count,
                              Kokkos::View<int *, ExecSpace> lambertian_indices,
                              Kokkos::View<int *, ExecSpace> missed_indices,
                              Kokkos::View<Hit *, ExecSpace> hits,
                              Kokkos::View<Lambertian *, ExecSpace> lambertians,
                              Kokkos::View<Ray *, ExecSpace> old_rays,
                              Kokkos::View<Ray *, ExecSpace> new_rays,
                              int global_seed) {

    int base_samples = 1;
    int extra_count = 0;
    int stride = 1;

    if (lambertian_count > 0 && missed_count > 0) {
        base_samples = 1 + (missed_count / lambertian_count);
        extra_count = missed_count % lambertian_count;

        if (extra_count > 0) {
            stride = lambertian_count / extra_count;
            if (stride == 0)
                stride = 1;
        }
    }
    Kokkos::parallel_for(
        "lambertian_samples",
        Kokkos::RangePolicy<ExecSpace>(0, lambertian_count),
        KOKKOS_LAMBDA(int i) {
            // simple per-thread XorShift RNG
            uint64_t state = global_seed + i * 0x9E3779B97F4A7C15ULL;
            auto rand = [&]() {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                return float(state & 0xFFFFFFFF) / float(0xFFFFFFFF);
            };

            int hit_index = lambertian_indices(i);
            Hit &hit = hits(hit_index);
            Lambertian mat = lambertians(hit.materialId);

            int my_samples = base_samples;
            int extra_sample_idx = -1;
            if (extra_count > 0) {
                if ((i % stride) == 0) {
                    int extras_before_me = i / stride;
                    if (extras_before_me < extra_count) {
                        my_samples++;
                        extra_sample_idx =
                            (base_samples - 1) * lambertian_count +
                            extras_before_me;
                    }
                }
            }

            float energy_scale = 1.0f / float(my_samples);

            // Compute specular reflection direction
            Vec3 incident = old_rays(hit_index).direction;
            float dot_n = dot(incident, hit.normal);
            Vec3 reflect_dir = incident - hit.normal * (2.0f * dot_n);
            reflect_dir = reflect_dir.normalized();

            // Helper lambda to generate direction based on specularity
            auto generate_direction = [&]() -> Vec3 {
                float specularity = mat.specularity; // Assumes you add this to
                                                     // Lambertian struct
                if (rand() < specularity) {
                    // Specular reflection with optional roughness
                    float roughness =
                        0.0f; // Optional: add for glossy reflections
                    if (roughness > 0.0f) {
                        // Add some random perturbation for glossy specular
                        float z = rand() * 2.0f - 1.0f;
                        float a = rand() * 2.0f * 3.14159265f;
                        float r = sqrtf(1.0f - z * z) * roughness;
                        Vec3 perturb =
                            Vec3(r * cosf(a), r * sinf(a), z * roughness);
                        return (reflect_dir + perturb).normalized();
                    }
                    return reflect_dir;
                } else {
                    // Diffuse
                    return mat.random_diffuse_direction(hit.normal, rand);
                }
            };

            Vec3 dir = generate_direction();
            new_rays(hit_index).origin = hit.position + hit.normal * 0.001f;
            new_rays(hit_index).direction = dir;
            new_rays(hit_index).inv_dir =
                Vec3(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);
            new_rays(hit_index).color =
                mat.diffuse_color(old_rays(hit_index).color) * energy_scale;
            new_rays(hit_index).pixelIndex = old_rays(hit_index).pixelIndex;

            for (int s = 1; s < base_samples; s++) {
                int out_idx = missed_indices(i + (s - 1) * lambertian_count);
                Vec3 d = generate_direction();
                new_rays(out_idx).origin = hit.position + hit.normal * 0.001f;
                new_rays(out_idx).direction = d;
                new_rays(out_idx).inv_dir =
                    Vec3(1.0f / d.x, 1.0f / d.y, 1.0f / d.z);
                new_rays(out_idx).color =
                    mat.diffuse_color(old_rays(hit_index).color) * energy_scale;
                new_rays(out_idx).pixelIndex = old_rays(hit_index).pixelIndex;
            }

            if (extra_sample_idx >= 0) {
                int out_idx = missed_indices(extra_sample_idx);
                Vec3 d = generate_direction();
                new_rays(out_idx).origin = hit.position + hit.normal * 0.001f;
                new_rays(out_idx).direction = d;
                new_rays(out_idx).inv_dir =
                    Vec3(1.0f / d.x, 1.0f / d.y, 1.0f / d.z);
                new_rays(out_idx).color =
                    mat.diffuse_color(old_rays(hit_index).color) * energy_scale;
                new_rays(out_idx).pixelIndex = old_rays(hit_index).pixelIndex;
            }
        });
    /*Kokkos::parallel_for(
        "lambertian_samples",
        Kokkos::RangePolicy<ExecSpace>(0, lambertian_count),
        KOKKOS_LAMBDA(int i) {
            // simple per-thread XorShift RNG
            uint64_t state = global_seed + i * 0x9E3779B97F4A7C15ULL;
            auto rand = [&]() {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                return float(state & 0xFFFFFFFF) / float(0xFFFFFFFF);
            };

            int hit_index = lambertian_indices(i);
            Hit &hit = hits(hit_index);
            Lambertian mat = lambertians(hit.materialId);

            int my_samples = base_samples;
            int extra_sample_idx = -1;

            if (extra_count > 0) {
                if ((i % stride) == 0) {
                    int extras_before_me = i / stride;
                    if (extras_before_me < extra_count) {
                        my_samples++;
                        extra_sample_idx =
                            (base_samples - 1) * lambertian_count +
                            extras_before_me;
                    }
                }
            }

            float energy_scale = 1.0f / float(my_samples);

            Vec3 dir = mat.random_diffuse_direction(hit.normal, rand);
            new_rays(hit_index).origin = hit.position + hit.normal * 0.001f;
            new_rays(hit_index).direction = dir;
            new_rays(hit_index).inv_dir =
                Vec3(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);
            new_rays(hit_index).color =
                mat.diffuse_color(old_rays(hit_index).color) * energy_scale;
            new_rays(hit_index).pixelIndex = old_rays(hit_index).pixelIndex;

            for (int s = 1; s < base_samples; s++) {
                int out_idx = missed_indices(i + (s - 1) * lambertian_count);
                Vec3 d = mat.random_diffuse_direction(hit.normal, rand);
                new_rays(out_idx).origin = hit.position + hit.normal * 0.001f;
                new_rays(out_idx).direction = d;
                new_rays(out_idx).inv_dir =
                    Vec3(1.0f / d.x, 1.0f / d.y, 1.0f / d.z);
                new_rays(out_idx).color =
                    mat.diffuse_color(old_rays(hit_index).color) * energy_scale;
                new_rays(out_idx).pixelIndex = old_rays(hit_index).pixelIndex;
            }

            if (extra_sample_idx >= 0) {
                int out_idx = missed_indices(extra_sample_idx);
                Vec3 d = mat.random_diffuse_direction(hit.normal, rand);
                new_rays(out_idx).origin = hit.position + hit.normal * 0.001f;
                new_rays(out_idx).direction = d;
                new_rays(out_idx).inv_dir =
                    Vec3(1.0f / d.x, 1.0f / d.y, 1.0f / d.z);
                new_rays(out_idx).color =
                    mat.diffuse_color(old_rays(hit_index).color) * energy_scale;
                new_rays(out_idx).pixelIndex = old_rays(hit_index).pixelIndex;
            }
        });*/
}

void handle_volumetric_scattering(Kokkos::View<Ray *, ExecSpace> new_rays,
                                  Kokkos::View<Ray *, ExecSpace> old_rays,
                                  Kokkos::View<Hit *, ExecSpace> hits,
                                  int ray_count, int global_seed) {
    Kokkos::parallel_for(
        "volumetric_scattering", Kokkos::RangePolicy<ExecSpace>(0, ray_count),
        KOKKOS_LAMBDA(int i) {
            Hit &hit = hits(i);

            uint64_t state = global_seed + i * 0x9E3779B97F4A7C15ULL;
            auto rand = [&]() {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                return float(state & 0xFFFFFFFF) / float(0xFFFFFFFF);
            };
            float max_scatter_distance = 20.0f;
            float scatter_base_prob = 0.01f;
            float t = hit.t;

            float distance = min(t, max_scatter_distance);
            float scatter_prob =
                scatter_base_prob * (distance / max_scatter_distance);

            if (rand() < scatter_prob) {
                float scatter_t = rand() * t;
                Vec3 scatter_point =
                    old_rays(i).origin + old_rays(i).direction * scatter_t;

                Vec3 sun_dir = Vec3(0.3f, -1.0f, 0.5f).normalized();

                float theta = acosf(1.0f - rand() * 0.2f);
                float phi = 2.0f * 3.14159265f * rand();
                float x = sinf(theta) * cosf(phi);
                float y = sinf(theta) * sinf(phi);
                float z = cosf(theta);
                Vec3 random_cone(x, y, z);

                Vec3 new_dir = (random_cone + sun_dir * 3.0f).normalized();
                new_rays(i).origin = scatter_point;
                new_rays(i).direction = new_dir;
                new_rays(i).inv_dir =
                    Vec3(1.0f / new_dir.x, 1.0f / new_dir.y, 1.0f / new_dir.z);
                new_rays(i).color = old_rays(i).color * 0.9f;
            }
        });
}

void bounce(Kokkos::View<Ray *, ExecSpace> new_rays,
            Kokkos::View<Ray *, ExecSpace> old_rays,
            Kokkos::View<Hit *, ExecSpace> hits,
            Kokkos::View<Mirror *, ExecSpace> mirrors,
            Kokkos::View<Lambertian *, ExecSpace> lambertians,
            Kokkos::View<Glass *, ExecSpace> glasses,
            Kokkos::View<Emissive *, ExecSpace> emissives,
            Kokkos::View<int *, ExecSpace> mirror_indices,
            Kokkos::View<int *, ExecSpace> lambertian_indices,
            Kokkos::View<int *, ExecSpace> glass_indices,
            Kokkos::View<int *, ExecSpace> emissive_indices,
            Kokkos::View<int *, ExecSpace> missed_indices, int mirror_count,
            int lambertian_count, int glass_count, int emissive_count,
            int missed_count, Kokkos::View<Vec3 *, ExecSpace> colors, int seed,
            bool last_bounce, int ray_count) {

    handle_emissive_hits(new_rays, old_rays, hits, emissive_indices, emissives,
                         colors, emissive_count);
    handle_missed_rays(new_rays, old_rays, hits, missed_indices, colors,
                       missed_count);

    if (last_bounce)
        return;
    if (mirror_count != 0) {
        handle_mirror_rays(new_rays, old_rays, hits, mirror_indices, mirrors,
                           mirror_count);
    }

    if (glass_count != 0) {
        handle_glass_rays(new_rays, old_rays, hits, glass_indices, glasses,
                          seed + 100000000, glass_count);
    }
    if (lambertian_count != 0) {
        spawn_lambertian_samples(lambertian_count, missed_count,
                                 lambertian_indices, missed_indices, hits,
                                 lambertians, old_rays, new_rays, seed);
    }

    /*handle_volumetric_scattering(new_rays, old_rays, hits, ray_count,
                                 seed + 200000000);*/
}

void normalize_colors(Kokkos::View<Vec3 *, ExecSpace> colors, int n, int spp) {
    Kokkos::parallel_for(
        "normalize_colors", Kokkos::RangePolicy<ExecSpace>(0, n),
        KOKKOS_LAMBDA(int i) { colors(i) = colors(i) / float(spp); });
}

void trace_once(int width, int height, Kokkos::View<Ray *, ExecSpace> rays,
                Kokkos::View<Ray *, ExecSpace> rays_new,
                Kokkos::View<Hit *, ExecSpace> hits,
                Kokkos::View<Vec3 *, ExecSpace> colors,
                Kokkos::View<Triangle *, ExecSpace> triangles,
                Kokkos::View<int *, ExecSpace> bvh_roots, int bvh_root_count,
                Kokkos::View<Mirror *, ExecSpace> mirrors,
                Kokkos::View<Glass *, ExecSpace> glasses,
                Kokkos::View<Lambertian *, ExecSpace> lambertians,
                Kokkos::View<Emissive *, ExecSpace> emissive,
                Kokkos::View<int *, ExecSpace> mirror_indices,
                Kokkos::View<int *, ExecSpace> lambertian_indices,
                Kokkos::View<int *, ExecSpace> glass_indices,
                Kokkos::View<int *, ExecSpace> emissive_indices,
                Kokkos::View<int *, ExecSpace> missed_indices,
                Kokkos::View<gpuBVHNode *, ExecSpace> bvh_nodes, int seed) {
    const int ray_count = width * height;

    Kokkos::View<int, ExecSpace> mirror_counter("mirror_counter");
    Kokkos::View<int, ExecSpace> lambertian_counter("lambertian_counter");
    Kokkos::View<int, ExecSpace> glass_counter("glass_counter");
    Kokkos::View<int, ExecSpace> emissive_counter("emissive_counter");
    Kokkos::View<int, ExecSpace> missed_counter("missed_counter");

    for (int bounce_i = 0; bounce_i < 3; bounce_i++) {
        Kokkos::deep_copy(mirror_counter, 0);
        Kokkos::deep_copy(lambertian_counter, 0);
        Kokkos::deep_copy(glass_counter, 0);
        Kokkos::deep_copy(emissive_counter, 0);
        Kokkos::deep_copy(missed_counter, 0);

        intersect_triangles(rays, hits, bvh_nodes, bvh_roots, bvh_root_count,
                            triangles, ray_count);

        caracterize_hits(hits, mirror_indices, lambertian_indices,
                         glass_indices, emissive_indices, missed_indices,
                         mirror_counter, lambertian_counter, glass_counter,
                         emissive_counter, missed_counter, ray_count);

        int mirror_h, lambertian_h, glass_h, emissive_h, missed_h;
        Kokkos::deep_copy(mirror_h, mirror_counter);
        Kokkos::deep_copy(lambertian_h, lambertian_counter);
        Kokkos::deep_copy(glass_h, glass_counter);
        Kokkos::deep_copy(emissive_h, emissive_counter);
        Kokkos::deep_copy(missed_h, missed_counter);
        /*printf("Bounce %d: Mirrors=%d, Lambertians=%d, Glasses=%d, "
               "Emissive=%d, Missed=%d\n",
               bounce_i, mirror_h, lambertian_h, glass_h, emissive_h,
               missed_h - emissive_h);*/

        bounce(rays_new, rays, hits, mirrors, lambertians, glasses, emissive,
               mirror_indices, lambertian_indices, glass_indices,
               emissive_indices, missed_indices, mirror_h, lambertian_h,
               glass_h, emissive_h, missed_h, colors,
               seed * (bounce_i * 123 + 1123123), bounce_i == 50, ray_count);
        Kokkos::fence();
        std::swap(rays, rays_new);
    }
}
