
#include "bitunpack_cuda.hpp"
#include "macros.hpp"
#include <stdexcept>

constexpr uint32_t THREADS_PER_BLOCK = 256;

template <typename type>
__global__ void naive_unpack_kernel(
    uint8_t *bitpacked,
    type *unpacked) {}

template <>
__global__ void naive_unpack_kernel<bool>(
    uint8_t *bitpacked,
    bool *unpacked)
{
    // A kernel to serve as a baseline for (bad) performance

    const uint32_t flat_index = threadIdx.x + THREADS_PER_BLOCK * blockIdx.x;
    const uint32_t write_offset = 8 * THREADS_PER_BLOCK * blockIdx.x;
    const uint32_t local_index = threadIdx.x;

    // Read in 8 values with one uint8
    uint8_t read = bitpacked[flat_index];

#pragma unroll
    for (uint32_t i = 0; i < 8; ++i)
    {
        unpacked[write_offset + 8 * local_index + i] = static_cast<bool>((read >> (7 - i)) & 1);
    }
}

template <>
__global__ void naive_unpack_kernel<uint8_t>(
    uint8_t *bitpacked,
    uint8_t *unpacked)
{
    // A kernel to serve as a baseline for (bad) performance

    const uint32_t flat_index = threadIdx.x + THREADS_PER_BLOCK * blockIdx.x;
    const uint32_t write_offset = 8 * THREADS_PER_BLOCK * blockIdx.x;
    const uint32_t local_index = threadIdx.x;

    // Read in 8 values with one uint8
    uint8_t read = bitpacked[flat_index];

#pragma unroll
    for (uint32_t i = 0; i < 8; ++i)
    {
        unpacked[write_offset + 8 * local_index + i] = ((read >> (7 - i)) & 1);
    }
}

template <typename type>
__global__ void unpack_kernel(
    uint8_t *bitpacked,
    type *unpacked) {}

template <>
__global__ void unpack_kernel<bool>(
    uint8_t *bitpacked,
    bool *unpacked)
{
    const uint32_t flat_index = threadIdx.x + THREADS_PER_BLOCK * blockIdx.x;
    const uint32_t write_offset = 8 * THREADS_PER_BLOCK * blockIdx.x;
    const uint32_t local_index = threadIdx.x;

    // Read in 8 values with one uint8
    uint8_t read = bitpacked[flat_index];

    // Convert to a 64 bit value so we can later reinterpret at 8 8-bit values.
    // CUDA is little-endian, so the binary representation that we read in:
    //      ABCDEFGH
    // needs to be converted to:
    //      0000000H 0000000G 0000000F 0000000E 0000000D 0000000C 0000000B 0000000A
    // in order to later become 8 bools like:
    //      A B C D E F G H
    constexpr uint64_t bit = 1;
    uint64_t RESULTS = static_cast<uint64_t>((read >> 7) & bit);

#pragma unroll
    for (uint32_t i = 1; i < 8; ++i)
    {
        RESULTS += (static_cast<uint64_t>(read) << (9 * i - 7)) & (bit << (8 * i));
    }

    // Write results to shared memory
    __shared__ uint8_t SHARED_RESULTS[THREADS_PER_BLOCK * 8];
    reinterpret_cast<uint64_t *>(SHARED_RESULTS)[local_index] = RESULTS;
    __syncthreads();

// Write to the unpacked tensor in a loop
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i)
    {
        uint32_t index_this_loop = local_index + THREADS_PER_BLOCK * i;
        unpacked[write_offset + index_this_loop] = static_cast<bool>(SHARED_RESULTS[index_this_loop]);
    }
}

template <>
__global__ void unpack_kernel<uint8_t>(
    uint8_t *bitpacked,
    uint8_t *unpacked)
{
    const uint32_t flat_index = threadIdx.x + THREADS_PER_BLOCK * blockIdx.x;
    const uint32_t write_offset = 8 * THREADS_PER_BLOCK * blockIdx.x;
    const uint32_t local_index = threadIdx.x;

    // Read in 8 values with one uint8
    uint8_t read = bitpacked[flat_index];

    // Convert to a 64 bit value so we can later reinterpret as 8 8-bit values.
    // CUDA is little-endian, so the binary representation that we read in:
    //      ABCDEFGH
    // needs to be converted to:
    //      0000000H 0000000G 0000000F 0000000E 0000000D 0000000C 0000000B 0000000A
    // in order to later become 8 bools like:
    //      A B C D E F G H
    constexpr uint64_t bit = 1;
    uint64_t RESULTS = static_cast<uint64_t>((read >> 7) & bit);

#pragma unroll
    for (uint32_t i = 1; i < 8; ++i)
    {
        RESULTS += (static_cast<uint64_t>(read) << (9 * i - 7)) & (bit << (8 * i));
    }

    // Write results to shared memory
    __shared__ uint8_t SHARED_RESULTS[THREADS_PER_BLOCK * 8];
    reinterpret_cast<uint64_t *>(SHARED_RESULTS)[local_index] = RESULTS;
    __syncthreads();

// Write to the unpacked tensor in a loop
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i)
    {
        uint32_t index_this_loop = local_index + THREADS_PER_BLOCK * i;
        unpacked[write_offset + index_this_loop] = SHARED_RESULTS[index_this_loop];
    }
}

inline std::tuple<uint32_t, uint32_t> check_inputs(
    torch::Tensor bitpacked,
    int64_t N,
    int64_t M,
    at::ScalarType dtype)
{
    // Must be a 2-dimensional tensor
    if (bitpacked.ndimension() != 2)
    {
        throw std::invalid_argument("Input tensor must be 2-dimensional");
    }

    // Must have uint8 datatype
    if (bitpacked.dtype() != torch::kUInt8)
    {
        throw std::invalid_argument("Input tensor must have dtype uint8");
    }

    // The desired shape must be possible through slicing
    uint32_t P = bitpacked.size(0);
    uint32_t K = bitpacked.size(1);
    if ((P < N) || (8 * K < M))
    {
        throw std::invalid_argument("Desired tensor shape is not possible");
    }

    // Must be either bool or uint8 return dtype
    if ((dtype != torch::kBool) && (dtype != torch::kUInt8))
    {
        throw std::invalid_argument("Supported output types are bool and uint8");
    }

    return {P, K};
}

template <at::ScalarType scalar_t>
inline torch::Tensor initialize_output_tensor(
    torch::Device device,
    uint32_t P,
    uint32_t K,
    uint32_t writes_per_block)
{
    torch::Tensor unpacked;

    // Zero pad so that the output unpacked tensor has an element number which is a multiple
    // of the writes per block
    if ((P * 8 * K) % writes_per_block == 0)
    {
        unpacked = torch::zeros({P * 8 * K}, torch::TensorOptions().dtype(scalar_t).device(device)); // [P * 8K]
    }
    else
    {
        const uint32_t padding = writes_per_block - ((P * 8 * K) % writes_per_block);
        unpacked = torch::zeros({P * 8 * K + padding}, torch::TensorOptions().dtype(scalar_t).device(device)); // [P * 8K + pad]
    }

    return unpacked;
}

torch::Tensor bitunpack_2d_CUDA(
    torch::Tensor bitpacked, // [P, K]
    int64_t N,
    int64_t M,
    at::ScalarType dtype,
    int64_t kernel)
{
    using namespace torch::indexing;

    auto [P, K] = check_inputs(bitpacked, N, M, dtype);
    torch::Tensor unpacked;

    AT_DISPATCH_OUTPUT_BOOL_OR_UINT8(dtype, [&]
                                     {
        // Instantiate the output
        constexpr uint32_t writes_per_block = THREADS_PER_BLOCK * 8;
        unpacked = initialize_output_tensor<PT_DTYPE>(bitpacked.device(), P, K, writes_per_block);
    
        dim3 threads(THREADS_PER_BLOCK, 1, 1);
        const uint32_t bNum = (P * 8 * K) / writes_per_block + ((P * 8 * K) % writes_per_block != 0 ? 1 : 0);
        dim3 blocks(bNum, 1, 1);
    
        uint8_t* bitpacked_ptr = bitpacked.data_ptr<uint8_t>();
        _TYPE* unpacked_ptr = unpacked.data_ptr<_TYPE>();

        if (kernel == 0)
        {
            unpack_kernel<_TYPE><<<blocks, threads>>>(bitpacked_ptr, unpacked_ptr);
        }
        else
        {
            naive_unpack_kernel<_TYPE><<<blocks, threads>>>(bitpacked_ptr, unpacked_ptr);
        } });

    // Reshape and slice off unwanted bits
    unpacked = unpacked.index({Slice(0, P * 8 * K)}).reshape({P, -1}).index({Slice(0, N), Slice(0, M)}).contiguous(); // [N, M]

    return unpacked; // [N, M]
}