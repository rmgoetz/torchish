
#include "raycast.hpp"
#include "../vector_f.cuh"

__global__ void raycast_kernel(
    torch::PackedTensorAccessor32<float, 3> origins,
    torch::PackedTensorAccessor32<float, 3> directions,
    torch::PackedTensorAccessor32<float, 2> vertices,
    torch::PackedTensorAccessor32<int64_t, 2> faces,
    torch::PackedTensorAccessor32<int64_t, 1> faces_per_batch,
    torch::PackedTensorAccessor32<int64_t, 1> face_index_start,
    torch::PackedTensorAccessor32<float, 2> distances,
    torch::PackedTensorAccessor32<float, 3> normals,
    const int B,
    const int R,
    const int F,
    const int threads_per_block)
{
    // The index for this thread within the block
    const uint32_t thread_index = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);

    // The first block dimension corresponds to the batch index this thread works for
    const uint32_t batch_index = blockIdx.x;

    // The second block dimension corresponds to a set of threads_per_block rays
    const uint32_t ray_index = thread_index + threads_per_block * blockIdx.y;

    // Likely not much of a performance impact, but leverage a small shared memory
    // SHARED[0] -> first face index for this batch
    // SHARED[1] -> number of faces in this batch
    __shared__ int64_t SHARED[2];
    if (thread_index == 0)
    {
        SHARED[thread_index] = face_index_start[batch_index];
    }
    else if (thread_index == 1)
    {
        SHARED[thread_index] = faces_per_batch[batch_index];
    }
    __syncthreads();

    // Do nothing for the thread outside of the ray bounds
    if (ray_index >= R)
    {
        return;
    }

    //
    float *origin_ptr = &origins[batch_index][ray_index][0];
    float *direction_ptr = &directions[batch_index][ray_index][0];
    cudVec3D_f origin = cudVec3D_f(origin_ptr);
    cudVec3D_f direction = cudVec3D_f(direction_ptr);

    //
    const int64_t face_start = SHARED[0];
    const int64_t face_stop = face_start + SHARED[1];

    //
    float infinity = std::numeric_limits<float>::infinity();
    float min_distance = infinity;

    //
    float eps1 = 1e-7;
    float eps2 = 1e-4;

    //
    float closest_face_normal_x;
    float closest_face_normal_y;
    float closest_face_normal_z;

    //
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

    //
    for (int face_index = face_start; face_index < face_stop; face_index++)
    {
        //
        float *vertex_0_ptr = &vertices[faces[face_index][0]][0];
        float *vertex_1_ptr = &vertices[faces[face_index][1]][0];
        float *vertex_2_ptr = &vertices[faces[face_index][2]][0];
        cudVec3D_f vertex_0 = cudVec3D_f(vertex_0_ptr);
        cudVec3D_f vertex_1 = cudVec3D_f(vertex_1_ptr);
        cudVec3D_f vertex_2 = cudVec3D_f(vertex_2_ptr);

        //
        edge_0_1 = vertex_1 - vertex_0;
        edge_0_2 = vertex_2 - vertex_0;
        face_normal = edge_0_1.cross(edge_0_2);

        //
        n_dot_rayd = face_normal.dot(direction);
        if (fabs(n_dot_rayd) < eps1)
        {
            continue;
        }

        //
        t = (vertex_0 - origin).dot(face_normal) / n_dot_rayd;

        //
        if (t < -eps1)
        {
            continue;
        }

        //
        rayd_cross_shifto = direction.cross(origin - vertex_0);
        scale = direction.dot(face_normal);
        beta = edge_0_2.dot(rayd_cross_shifto) / scale;
        gamma = -edge_0_1.dot(rayd_cross_shifto) / scale;
        alpha = 1 - beta - gamma;

        //
        bool in_face = (alpha > -eps2) && (alpha < 1 + eps2) && (beta > -eps2) && (beta < 1 + eps2) && (gamma > -eps2) && (gamma < 1 + eps2);

        //
        if (!in_face)
        {
            continue;
        }

        //
        if (t < min_distance)
        {
            //
            min_distance = t;

            //
            face_normal = face_normal.normalize();

            //
            const int sign = (face_normal.dot(direction) > 0) ? -1 : 1;
            closest_face_normal_x = face_normal[0] * sign;
            closest_face_normal_y = face_normal[1] * sign;
            closest_face_normal_z = face_normal[2] * sign;
        }
    }

    //
    if (min_distance < infinity)
    {
        distances[batch_index][ray_index] = min_distance;
        normals[batch_index][ray_index][0] = closest_face_normal_x;
        normals[batch_index][ray_index][1] = closest_face_normal_y;
        normals[batch_index][ray_index][2] = closest_face_normal_z;
    }
}

std::vector<torch::Tensor> raycast_CUDA(
    torch::Tensor origins,      // [B, R, 3 (x, y, z)]
    torch::Tensor directions,   // [B, R, 3 (x, y, z)]
    torch::Tensor vertices,     // [V, 3 (x, y, z)]
    torch::Tensor faces,        // [F, 3 (v0, v1, v2)]
    torch::Tensor vertex_batch) // [V] consecutive and sorted
{
    using namespace torch::indexing;

    // The number of batches of rays
    const uint32_t B = origins.size(0);

    // The number of rays
    const uint32_t R = origins.size(1);

    // The number of total triangulated mesh facets across all batches
    const uint32_t F = faces.size(0);

    // The tensor corresponding each facet to a batch index
    torch::Tensor face_batch = vertex_batch.index({faces.index({Slice(), 0})}).contiguous(); // [F]

    // The number of faces in each batch
    std::tuple<at::Tensor, at::Tensor, at::Tensor> unique_result = at::_unique2(face_batch, true, false, true);
    torch::Tensor faces_per_batch = std::get<2>(unique_result); // [B]

    // The index of the first face for each batch index
    torch::Tensor face_index_start = torch::cat({torch::zeros({1}, faces_per_batch.options()),
                                                 torch::cumsum(faces_per_batch, 0).index({Slice{0, B - 1}})},
                                                0); // [B]

    // Instantiate the distances tensor and
    float inf = std::numeric_limits<float>::infinity();
    torch::Tensor distances = torch::full({B, R}, inf, origins.options());  // [B, R]
    torch::Tensor normals = torch::full({B, R, 3}, inf, origins.options()); // [B, R, 3 (x, y, z)]

    //
    dim3 threads(256, 1, 1);
    const int threads_per_block = threads.x * threads.y * threads.z;
    const int R_blocks = R / threads_per_block + ((R % threads_per_block != 0) ? 1 : 0);
    dim3 blocks(B, R_blocks, 1);

    // Launch the kernel
    raycast_kernel<<<blocks, threads>>>(
        origins.packed_accessor32<float, 3>(),
        directions.packed_accessor32<float, 3>(),
        vertices.packed_accessor32<float, 2>(),
        faces.packed_accessor32<int64_t, 2>(),
        faces_per_batch.packed_accessor32<int64_t, 1>(),
        face_index_start.packed_accessor32<int64_t, 1>(),
        distances.packed_accessor32<float, 2>(),
        normals.packed_accessor32<float, 3>(),
        B,
        R,
        F,
        threads_per_block);

    return {distances, normals}; // [B, R], [B, R, 3 (x, y, z)]
}
