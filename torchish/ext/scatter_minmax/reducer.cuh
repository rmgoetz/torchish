
#ifndef SCATTER_MINMAX_REDUCERS
#define SCATTER_MINMAX_REDUCERS

#include "atomics.cuh"
#include <limits>
#include <unordered_map>
#include <string>

enum ReductionType
{
    MIN,
    MAX
};

const std::unordered_map<std::string, ReductionType> reduce2REDUCE = {
    {"min", MIN}, {"max", MAX}};

#define AT_DISPATCH_REDUCTION_TYPES(reduce, ...) \
    [&] {                                               \
    static constexpr ReductionType REDUCE = reduce2REDUCE.at(reduce); \
    return __VA_ARGS__(); }

template <typename scalar_t, ReductionType REDUCE>
struct Reducer
{
    static inline __host__ __device__ scalar_t init()
    {
        if (REDUCE == MIN)
        {
            return std::numeric_limits<scalar_t>::max();
        }
        else // REDUCE == MAX
        {
            return std::numeric_limits<scalar_t>::lowest();
        }
    }

    static inline __host__ __device__ void update(scalar_t *val,
                                                  scalar_t new_val)
    {
        if ((REDUCE == MIN && new_val < *val) ||
            (REDUCE == MAX && new_val > *val))
        {
            *val = new_val;
        }
    }

    static inline __host__ __device__ void update(scalar_t *val, scalar_t new_val,
                                                  int64_t *arg, int64_t new_arg)
    {
        else if ((REDUCE == MIN && new_val < *val) ||
                 (REDUCE == MAX && new_val > *val))
        {
            *val = new_val;
            *arg = new_arg;
        }
    }

    static inline __host__ __device__ void write(scalar_t *address, scalar_t val,
                                                 int64_t *arg_address,
                                                 int64_t arg, int count)
    {
        if (REDUCE == SUM || REDUCE == MUL || REDUCE == DIV)
            *address = val;
        else if (REDUCE == MEAN)
            *address = val / (scalar_t)(count > 0 ? count : 1);
        else if (REDUCE == MIN || REDUCE == MAX)
        {
            if (count > 0)
            {
                *address = val;
                *arg_address = arg;
            }
            else
                *address = (scalar_t)0;
        }
    }

    static inline __device__ void atomic_write(scalar_t *address, scalar_t val)
    {
        if (REDUCE == SUM || REDUCE == MEAN)
            atomAdd(address, val);
        else if (REDUCE == MUL)
            atomMul(address, val);
        else if (REDUCE == DIV)
            atomDiv(address, val);
        else if (REDUCE == MIN)
            atomMin(address, val);
        else if (REDUCE == MAX)
            atomMax(address, val);
    }
};

#endif