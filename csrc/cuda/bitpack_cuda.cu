
#include "bitpack_cuda.hpp"
#include <stdexcept>

// The number of threads to launch a block with in the lineread kernel
constexpr uint32_t THREADS_PER_BLOCK = 256;

// The number of times a thread will write to the output in the lineread kernel.
// Testing seems to suggest that 1 is optimal, but this may not hold true for all
// GPU devices.
constexpr uint32_t WRITES_PER_THREAD = 1;

__global__ void bitpack_kernel_lineread(
    const bool *input,
    uint8_t *compact,
    const int num_write_elems)
{
    // The index for this thread within the block
    const uint32_t thread_index = threadIdx.x; // + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);

    // The shared memory where we'll store partially compacted uint8s
    __shared__ uint8_t PARTIAL_SHARE[THREADS_PER_BLOCK * 2];

    // Each block is responsible for 8 * threads_per_block * writes_per_thread number of bool reads from the
    // input tensor, however, they are read in 32 bits at a time
    const uint32_t read_offset = (8 * THREADS_PER_BLOCK * WRITES_PER_THREAD * blockIdx.x) / 4;

    // A pointer which will treat the input as not a tensor of byte-sized bools, but instead a
    // tensor of int32, so that we can read more data in a single warp
    const uint32_t *grouped_input_ptr = reinterpret_cast<const uint32_t *>(input) + read_offset;

    // Loop over the desired number of writes per thread for the block
    for (uint32_t write_number = 0; write_number < WRITES_PER_THREAD; ++write_number)
    {
        // Each block is responsible for thread_per_block number of writes to the compact tensor
        uint32_t write_index = thread_index +
                               (write_number + WRITES_PER_THREAD * blockIdx.x) * THREADS_PER_BLOCK;

        // A flag which is true when our thread corresponds to a write out of bounds
        bool past_tensor_edge = write_index >= num_write_elems;

        // Because bools in PyTorch are single byte, and a warp can access up to 128 bytes in a single
        // memory access, the number of warps we need is:
        //      total_bool_reads / 128 bools_read_per_warp
        //          = (threads_per_block * 8) / 128
        //          = threads_per_block / 16
        // And because the number of warps per block is:
        //      threads_per_block / 32
        // we conclude that we need two loops of accesses to read all of the input values for this block
        for (uint32_t access_index = 0; access_index < 2; ++access_index)
        {
            // Read the 4-bool group as an int32
            uint32_t grouped_vals = grouped_input_ptr[thread_index];

            // CUDA is little-endian, so when we reinterpret cast our boolean array to uint32, the bytes arrayed as:
            //      0000000A 0000000B 0000000C 0000000D
            // will be arrayed in the binary integer value as:
            //      0000000D 0000000C 0000000B 0000000A
            // which we want to bitshift into:
            //      0000ABCD
            uint8_t val = 0;
            for (int bit = 0; bit < 3; ++bit)
            {
                uint32_t shift = 24 - (9 * bit);
                val |= static_cast<uint8_t>(grouped_vals >> shift);
            }
            val |= static_cast<uint8_t>(grouped_vals << 3);

            // Write to shared memory
            PARTIAL_SHARE[thread_index + THREADS_PER_BLOCK * access_index] = val;

            // Update the read pointer
            grouped_input_ptr += THREADS_PER_BLOCK;
        }
        __syncthreads();

        // Read the pairs of intermediate values. Again because of little-endianness our two bytes in memory:
        //      0000ABCD 0000EFGH
        // are going to be interpreted in binary as:
        //      0000EFGH 0000ABCD
        // which we want to bitpack into a single uint8 as:
        //      ABCDEFGH
        const uint16_t *grouped_share_ptr = reinterpret_cast<const uint16_t *>(PARTIAL_SHARE);
        const uint16_t left_right_pairs = grouped_share_ptr[thread_index];
        uint8_t write_val = (left_right_pairs << 4) + (left_right_pairs >> 8);

        // Write the compactified value
        if (!past_tensor_edge)
        {
            compact[write_index] = write_val;
        }
    }
}

__global__ void bitpack_kernel_vectorized(
    bool *input,
    uint8_t *compact)
{
    // Thread's index in the full scope of the kernel
    const uint32_t flat_index = threadIdx.x + THREADS_PER_BLOCK * blockIdx.x;

    // Each thread will load 16 bytes, which is 16 bools, which will become 2 uints in the compacted space
    uint4 vec = reinterpret_cast<uint4 *>(input)[flat_index];

    // Initialize our write values
    uint8_t write_vals[2] = {0, 0};

    // CUDA is litte-endian, so our two uint32s with bytes like:
    //      0000000A 0000000B 0000000C 0000000D      0000000E 0000000F 0000000G 0000000H
    // will have binary interpretations as:
    //      0000000D 0000000C 0000000B 0000000A      0000000H 0000000G 0000000F 0000000E
    // which we want to compact into:
    //      ABCDEFGH

    write_vals[0] += (vec.x << 7) + (vec.y << 3);
    write_vals[1] += (vec.z << 7) + (vec.w << 3);

    for (uint32_t i = 0; i < 3; ++i)
    {
        write_vals[0] += (vec.x >> (2 + (9 * i))) + (vec.y >> (6 + (9 * i)));
        write_vals[1] += (vec.z >> (2 + (9 * i))) + (vec.w >> (6 + (9 * i)));
    }

    // Pack the two writes as a single uint16 and write to the output
    uint16_t packed = (static_cast<uint16_t>(write_vals[1]) << 8) | static_cast<uint16_t>(write_vals[0]);
    reinterpret_cast<uint16_t *>(compact)[flat_index] = packed;
}

torch::Tensor bitpack_2d_CUDA(torch::Tensor input, int64_t kernel = 0)
{
    using namespace torch::indexing;

    if (input.ndimension() != 2)
    {
        throw std::invalid_argument("Input tensor must be 2-dimensional");
    }

    if (input.dtype() != torch::kBool)
    {
        throw std::invalid_argument("Input tensor must have dtype bool");
    }

    uint32_t N = input.size(0);
    uint32_t M = input.size(1);

    uint32_t K = M / 8 + ((M % 8 == 0) ? 0 : 1);

    torch::Tensor compact = torch::zeros({N, K}, torch::TensorOptions().dtype(torch::kUInt8).device(input.device())); // [N, K]

    // Pad the input tensor with zeros so that the column dimension is a multiple of 8
    if (M % 8 != 0)
    {
        input = torch::cat({input, torch::zeros({N, 8 - (M % 8)}, input.options())}, -1).contiguous(); // [N, 8*K]
    }

    if (kernel == 0)
    {
        const uint32_t writes_per_block = THREADS_PER_BLOCK * WRITES_PER_THREAD;
        dim3 threads(256, 1, 1);
        const uint32_t bNum = (N * K) / writes_per_block + (((N * K) % writes_per_block) != 0 ? 1 : 0);
        dim3 blocks(bNum, 1, 1);

        auto input_ptr = input.data_ptr<bool>();
        auto compact_ptr = compact.data_ptr<uint8_t>();

        bitpack_kernel_lineread<<<blocks, threads>>>(input_ptr, compact_ptr, N * K);

    }
    else
    {
        // Reshape to 1-dimensional, then pad the output tensor to be a multiple of the number of writes per block
        input = input.view({-1});     // [N * (8*K)]
        compact = compact.view({-1}); // [N * K]
        const uint32_t writes_per_block = THREADS_PER_BLOCK * 2;
        if ((N * K) % (writes_per_block) != 0)
        {
            compact = torch::cat({compact, torch::zeros({writes_per_block - ((N * K) % writes_per_block)}, compact.options())}, 0).contiguous(); // [N * K + pad]
        }
        dim3 threads(256, 1, 1);
        const uint32_t bNum = (N * K) / writes_per_block + (((N * K) % writes_per_block) != 0 ? 1 : 0);
        dim3 blocks(bNum, 1, 1);

        auto input_ptr = input.data_ptr<bool>();
        auto compact_ptr = compact.data_ptr<uint8_t>();

        bitpack_kernel_vectorized<<<blocks, threads>>>(input_ptr, compact_ptr);

        // Slice away the padding on the output and reshape
        compact = compact.index({Slice(0, N * K)}).reshape({N, K}); // [N, K]
    }

    return compact; // [N, K]
}