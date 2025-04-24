
#include "raycast_cuda.hpp"
#include "vector_f.cuh"

__global__ void raycast_kernel_nb_simple(
    torch::PackedTensorAccessor32<float, 2> origins,           // [R, 3 (x, y, z)]
    torch::PackedTensorAccessor32<float, 2> directions,        // [R, 3 (x, y, z)]
    torch::PackedTensorAccessor32<float, 3> vertices_unrolled, // [F, 3 (v0, v1, v2), 3 (x, y, z)]
    torch::PackedTensorAccessor32<float, 1> distances,         // [R]
    torch::PackedTensorAccessor32<float, 2> normals,           // [R, 3 (x, y, z)]
    const int R,
    const int F,
    const int threads_per_block)
{
    // The index for this thread within the block
    const uint32_t thread_index = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);

    // The first block dimension corresponds to a set of threads_per_block rays
    const uint32_t ray_index = thread_index + threads_per_block * blockIdx.x;

    // Do nothing for the thread outside of the ray bounds
    if (ray_index >= R)
    {
        return;
    }

    // The origin and direction vectors for this thread
    float *origin_ptr = &origins[ray_index][0];
    float *direction_ptr = &directions[ray_index][0];
    cudVec3D_f origin = cudVec3D_f(origin_ptr);
    cudVec3D_f direction = cudVec3D_f(direction_ptr);

    // Initialize the minimum distance to infinity
    float infinity = std::numeric_limits<float>::infinity();
    float min_distance = infinity;

    // Tolerances for determining orientation and facet inclusion, respectively
    float eps1 = 1e-7;
    float eps2 = 1e-4;

    // Declare the facet normal components
    float closest_face_normal_x;
    float closest_face_normal_y;
    float closest_face_normal_z;

    // Declare some convenient quantities
    cudVec3D_f edge_0_1;
    cudVec3D_f edge_0_2;
    cudVec3D_f face_normal;
    cudVec3D_f rayd_cross_shifto;
    float n_dot_rayd;
    float t;
    float scale;
    float alpha;
    float beta;
    float gamma;

    // Loop over every facet in the batch
    for (int face_index = 0; face_index < F; face_index++)
    {
        // The 3D vertices defining the face
        float *vertex_0_ptr = &vertices_unrolled[face_index][0][0];
        float *vertex_1_ptr = &vertices_unrolled[face_index][1][0];
        float *vertex_2_ptr = &vertices_unrolled[face_index][2][0];
        cudVec3D_f vertex_0 = cudVec3D_f(vertex_0_ptr);
        cudVec3D_f vertex_1 = cudVec3D_f(vertex_1_ptr);
        cudVec3D_f vertex_2 = cudVec3D_f(vertex_2_ptr);

        // Three vectors characterizing the face
        edge_0_1 = vertex_1 - vertex_0;
        edge_0_2 = vertex_2 - vertex_0;
        face_normal = edge_0_1.cross(edge_0_2);

        // No collision if the face normal is perpendicular to the ray direction
        n_dot_rayd = face_normal.dot(direction);
        if (fabs(n_dot_rayd) < eps1)
        {
            continue;
        }

        // The signed distance from the ray origin to the plane of the face
        t = (vertex_0 - origin).dot(face_normal) / n_dot_rayd;

        // A negative distance means the ray points away from the face, so no collision
        if (t < -eps1)
        {
            continue;
        }

        // Determine whether the ray cast to the face plane falls within the face
        rayd_cross_shifto = direction.cross(origin - vertex_0);
        scale = direction.dot(face_normal);
        beta = edge_0_2.dot(rayd_cross_shifto) / scale;
        gamma = -edge_0_1.dot(rayd_cross_shifto) / scale;
        alpha = 1 - beta - gamma;
        bool in_face = (alpha > -eps2) && (alpha < 1 + eps2) && (beta > -eps2) && (beta < 1 + eps2) && (gamma > -eps2) && (gamma < 1 + eps2);

        // Outside of the face results in no collision
        if (!in_face)
        {
            continue;
        }

        // If t is less than the current minimum distance, update the minimum distance and face normal
        if (t < min_distance)
        {
            min_distance = t;

            // Make the normal vector unit normal
            face_normal = face_normal.normalize();

            // We choose the face normal to be opposing the ray direction
            const int sign = (face_normal.dot(direction) > 0) ? -1 : 1;

            closest_face_normal_x = face_normal[0] * sign;
            closest_face_normal_y = face_normal[1] * sign;
            closest_face_normal_z = face_normal[2] * sign;
        }
    }

    // If we've found a collision, write the results
    if (min_distance < infinity)
    {
        distances[ray_index] = min_distance;
        normals[ray_index][0] = closest_face_normal_x;
        normals[ray_index][1] = closest_face_normal_y;
        normals[ray_index][2] = closest_face_normal_z;
    }
}

__global__ void raycast_kernel_nb_exp1(
    torch::PackedTensorAccessor32<float, 2> origins,           // [R, 3 (x, y, z)]
    torch::PackedTensorAccessor32<float, 2> directions,        // [R, 3 (x, y, z)]
    torch::PackedTensorAccessor32<float, 3> vertices_unrolled, // [F, 3 (v0, v1, v2), 3 (x, y, z)]
    torch::PackedTensorAccessor32<float, 1> distances,         // [R]
    torch::PackedTensorAccessor32<float, 2> normals,           // [R, 3 (x, y, z)]
    const int R,
    const int F,
    const int threads_per_block)
{
    // The index for this thread within the block
    const uint32_t thread_index = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);

    // The first block dimension corresponds to a set of threads_per_block rays
    const uint32_t ray_index = thread_index + threads_per_block * blockIdx.x;

    // The origin and direction vectors for this thread
    float *origin_ptr = &origins[ray_index][0];
    float *direction_ptr = &directions[ray_index][0];
    cudVec3D_f origin = cudVec3D_f(origin_ptr);
    cudVec3D_f direction = cudVec3D_f(direction_ptr);

    // Initialize the minimum distance to infinity
    float infinity = std::numeric_limits<float>::infinity();
    float min_distance = infinity;

    // Tolerances for determining orientation and facet inclusion, respectively
    float eps1 = 1e-7;
    float eps2 = 1e-4;

    // Declare the facet normal components
    float closest_face_normal_x = infinity;
    float closest_face_normal_y = infinity;
    float closest_face_normal_z = infinity;

    // Declare some convenient quantities
    cudVec3D_f edge_0_1;
    cudVec3D_f edge_0_2;
    cudVec3D_f face_normal;
    cudVec3D_f rayd_cross_shifto;
    float n_dot_rayd;
    float t;
    float scale;
    float alpha;
    float beta;
    float gamma;
    bool is_perp;
    bool wrong_direction;
    bool in_face;

    // Loop over every facet in the batch
    for (int face_index = 0; face_index < F; face_index++)
    {
        // The 3D vertices defining the face
        float *vertex_0_ptr = &vertices_unrolled[face_index][0][0];
        float *vertex_1_ptr = &vertices_unrolled[face_index][1][0];
        float *vertex_2_ptr = &vertices_unrolled[face_index][2][0];
        cudVec3D_f vertex_0 = cudVec3D_f(vertex_0_ptr);
        cudVec3D_f vertex_1 = cudVec3D_f(vertex_1_ptr);
        cudVec3D_f vertex_2 = cudVec3D_f(vertex_2_ptr);

        // Three vectors characterizing the face
        edge_0_1 = vertex_1 - vertex_0;
        edge_0_2 = vertex_2 - vertex_0;
        face_normal = edge_0_1.cross(edge_0_2);

        // No collision if the face normal is perpendicular to the ray direction
        n_dot_rayd = face_normal.dot(direction);
        is_perp = fabs(n_dot_rayd) < eps1;

        // The signed distance from the ray origin to the plane of the face
        t = (vertex_0 - origin).dot(face_normal) / n_dot_rayd;

        // A negative distance means the ray points away from the face, so no collision
        wrong_direction = t < -eps1;

        // Determine whether the ray cast to the face plane falls within the face
        rayd_cross_shifto = direction.cross(origin - vertex_0);
        scale = direction.dot(face_normal);
        beta = edge_0_2.dot(rayd_cross_shifto) / scale;
        gamma = -edge_0_1.dot(rayd_cross_shifto) / scale;
        alpha = 1 - beta - gamma;

        // Outside of the face results in no collision
        bool in_face = (alpha > -eps2) && (alpha < 1 + eps2) && (beta > -eps2) && (beta < 1 + eps2) && (gamma > -eps2) && (gamma < 1 + eps2);

        // If t is less than the current minimum distance, update the minimum distance and face normal
        if ((t < min_distance) && (!is_perp) && (!wrong_direction) && (in_face))
        {
            min_distance = t;

            // Make the normal vector unit normal
            face_normal = face_normal.normalize();

            // We choose the face normal to be opposing the ray direction
            const int sign = (face_normal.dot(direction) > 0) ? -1 : 1;

            closest_face_normal_x = face_normal[0] * sign;
            closest_face_normal_y = face_normal[1] * sign;
            closest_face_normal_z = face_normal[2] * sign;
        }
    }

    // Write the results for valid rays
    if (ray_index < R)
    {
        distances[ray_index] = min_distance;
        normals[ray_index][0] = closest_face_normal_x;
        normals[ray_index][1] = closest_face_normal_y;
        normals[ray_index][2] = closest_face_normal_z;
    }
}

__global__ void raycast_kernel_nb_exp2(
    torch::PackedTensorAccessor32<float, 2> origins,           // [R, 3 (x, y, z)]
    torch::PackedTensorAccessor32<float, 2> directions,        // [R, 3 (x, y, z)]
    torch::PackedTensorAccessor32<float, 3> vertices_unrolled, // [F, 3 (v0, v1, v2), 3 (x, y, z)]
    torch::PackedTensorAccessor32<float, 1> distances,         // [R]
    torch::PackedTensorAccessor32<float, 2> normals,           // [R, 3 (x, y, z)]
    const int R,
    const int F,
    const int threads_per_block)
    {
        // The index for this thread within the block
        const uint32_t thread_index = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    
        // The first block dimension corresponds to a set of threads_per_block rays
        const uint32_t ray_index = thread_index + threads_per_block * blockIdx.x;
    
        // The origin and direction vectors for this thread
        float *origin_ptr = &origins[ray_index][0];
        float *direction_ptr = &directions[ray_index][0];
        cudVec3D_f origin = cudVec3D_f(origin_ptr);
        cudVec3D_f direction = cudVec3D_f(direction_ptr);
    
        // Initialize the minimum distance to infinity
        float infinity = std::numeric_limits<float>::infinity();
        float min_distance = infinity;
    
        // Tolerances for determining orientation and facet inclusion, respectively
        float eps1 = 1e-7;
        float eps2 = 1e-4;
    
        // Declare the facet normal components
        float closest_face_normal_x = infinity;
        float closest_face_normal_y = infinity;
        float closest_face_normal_z = infinity;
    
        // Declare some convenient quantities
        cudVec3D_f edge_0_1;
        cudVec3D_f edge_0_2;
        cudVec3D_f face_normal;
        cudVec3D_f rayd_cross_shifto;
        float n_dot_rayd;
        float t;
        float scale;
        float alpha;
        float beta;
        float gamma;
        bool is_perp;
        bool wrong_direction;
        bool in_face;
    
        // Loop over every facet in the batch
        for (int face_index = 0; face_index < F; face_index++)
        {
            // The 3D vertices defining the face
            float *vertex_0_ptr = &vertices_unrolled[face_index][0][0];
            float *vertex_1_ptr = &vertices_unrolled[face_index][1][0];
            float *vertex_2_ptr = &vertices_unrolled[face_index][2][0];
            cudVec3D_f vertex_0 = cudVec3D_f(vertex_0_ptr);
            cudVec3D_f vertex_1 = cudVec3D_f(vertex_1_ptr);
            cudVec3D_f vertex_2 = cudVec3D_f(vertex_2_ptr);
    
            // Three vectors characterizing the face
            edge_0_1 = vertex_1 - vertex_0;
            edge_0_2 = vertex_2 - vertex_0;
            face_normal = edge_0_1.cross(edge_0_2);
    
            // No collision if the face normal is perpendicular to the ray direction
            n_dot_rayd = face_normal.dot(direction);
            is_perp = fabs(n_dot_rayd) < eps1;
    
            // The signed distance from the ray origin to the plane of the face
            t = (vertex_0 - origin).dot(face_normal) / n_dot_rayd;
    
            // A negative distance means the ray points away from the face, so no collision
            wrong_direction = t < -eps1;
    
            // Determine whether the ray cast to the face plane falls within the face
            rayd_cross_shifto = direction.cross(origin - vertex_0);
            scale = direction.dot(face_normal);
            beta = edge_0_2.dot(rayd_cross_shifto) / scale;
            gamma = -edge_0_1.dot(rayd_cross_shifto) / scale;
            alpha = 1 - beta - gamma;
    
            // Outside of the face results in no collision
            bool in_face = (alpha > -eps2) && (alpha < 1 + eps2) && (beta > -eps2) && (beta < 1 + eps2) && (gamma > -eps2) && (gamma < 1 + eps2);
    
            // If t is less than the current minimum distance, update the minimum distance and face normal
            if ((t < min_distance) && (!is_perp) && (!wrong_direction) && (in_face))
            {
                min_distance = t;
    
                // Make the normal vector unit normal
                face_normal = face_normal.normalize();
    
                // We choose the face normal to be opposing the ray direction
                const int sign = (face_normal.dot(direction) > 0) ? -1 : 1;
    
                closest_face_normal_x = face_normal[0] * sign;
                closest_face_normal_y = face_normal[1] * sign;
                closest_face_normal_z = face_normal[2] * sign;
            }
        }
    
        // Write the results for valid rays
        if (ray_index < R)
        {
            distances[ray_index] = min_distance;
            normals[ray_index][0] = closest_face_normal_x;
            normals[ray_index][1] = closest_face_normal_y;
            normals[ray_index][2] = closest_face_normal_z;
        }
    }

std::vector<torch::Tensor> raycast_CUDA_nb(
    torch::Tensor origins,    // [R, 3 (x, y, z)]
    torch::Tensor directions, // [R, 3 (x, y, z)]
    torch::Tensor vertices,   // [V, 3 (x, y, z)]
    torch::Tensor faces,      // [F, 3 (v0, v1, v2)]
    int64_t kernel = 0)
{
    using namespace torch::indexing;

    // The number of rays
    const uint32_t R = origins.size(0);

    // The number of total triangulated mesh facets across all batches
    const uint32_t F = faces.size(0);

    // Unroll/unwrap the vertices into a single tensor
    torch::Tensor vertices_unrolled = vertices.index({faces.view({-1}), Slice()}).reshape({F, 3, 3}); // [F, 3 (v0, v1, v2), (x, y, z)]

    // Instantiate the distances tensor and
    float inf = std::numeric_limits<float>::infinity();
    torch::Tensor distances = torch::full({R}, inf, origins.options());  // [R]
    torch::Tensor normals = torch::full({R, 3}, inf, origins.options()); // [R, 3 (x, y, z)]

    // Normalize the directions tensor. In lieu of this we could always scale the distance results after
    // calling the kernel, but that method risks introducing more numerical error.
    torch::Tensor ndirections = directions / torch::linalg_norm(directions, 2, -1, true); // [R, 3 (x, y, z)]

    //
    dim3 threads(256, 1, 1);
    const int threads_per_block = threads.x * threads.y * threads.z;
    const int R_blocks = R / threads_per_block + ((R % threads_per_block != 0) ? 1 : 0);
    dim3 blocks(R_blocks, 1, 1);

    if (kernel == 0)
    {
        // The most straighforward kernel
        raycast_kernel_nb_simple<<<blocks, threads>>>(
            origins.packed_accessor32<float, 2>(),
            ndirections.packed_accessor32<float, 2>(),
            vertices_unrolled.packed_accessor32<float, 3>(),
            distances.packed_accessor32<float, 1>(),
            normals.packed_accessor32<float, 2>(),
            R,
            F,
            threads_per_block);
    }
    else if (kernel == 1)
    {
        // An experimental kernel
        raycast_kernel_nb_exp1<<<blocks, threads>>>(
            origins.packed_accessor32<float, 2>(),
            ndirections.packed_accessor32<float, 2>(),
            vertices_unrolled.packed_accessor32<float, 3>(),
            distances.packed_accessor32<float, 1>(),
            normals.packed_accessor32<float, 2>(),
            R,
            F,
            threads_per_block);
    }
    else
    {
        // An experimental kernel
        raycast_kernel_nb_exp2<<<blocks, threads>>>(
            origins.packed_accessor32<float, 2>(),
            ndirections.packed_accessor32<float, 2>(),
            vertices_unrolled.packed_accessor32<float, 3>(),
            distances.packed_accessor32<float, 1>(),
            normals.packed_accessor32<float, 2>(),
            R,
            F,
            threads_per_block);
    }

    return {distances, normals}; // [R], [R, 3 (x, y, z)]
}
