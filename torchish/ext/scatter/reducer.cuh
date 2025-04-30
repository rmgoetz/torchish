/**
 * Modified from the torch_scatter package.
 * 
 * Original license:
 * 
 * Copyright (c) 2020 Matthias Fey <matthias.fey@tu-dortmund.de>
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 * 
 */

#pragma once

#include <limits>
#include <map>
#include <string>

#include "atomics.cuh"

enum ReductionType { SUM, MEAN, MUL, DIV, MIN, MAX };

const std::map<std::string, ReductionType> reduce2REDUCE = {
    {"sum", SUM}, {"mean", MEAN}, {"mul", MUL},
    {"div", DIV}, {"min", MIN},   {"max", MAX},
};

#define AT_DISPATCH_REDUCTION_TYPES(reduce, ...)                               \
  [&] {                                                                        \
    switch (reduce2REDUCE.at(reduce)) {                                        \
    case SUM: {                                                                \
      static constexpr ReductionType REDUCE = SUM;                                        \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case MEAN: {                                                               \
      static constexpr ReductionType REDUCE = MEAN;                                       \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case MUL: {                                                                \
      static constexpr ReductionType REDUCE = MUL;                                        \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case DIV: {                                                                \
      static constexpr ReductionType REDUCE = DIV;                                        \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case MIN: {                                                                \
      static constexpr ReductionType REDUCE = MIN;                                        \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case MAX: {                                                                \
      static constexpr ReductionType REDUCE = MAX;                                        \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    }                                                                          \
  }()

template <typename scalar_t, ReductionType REDUCE> struct Reducer {
  static inline __host__ __device__ scalar_t init() {
    if (REDUCE == MUL || REDUCE == DIV)
      return (scalar_t)1;
    else if (REDUCE == MIN)
      return std::numeric_limits<scalar_t>::max();
    else if (REDUCE == MAX)
      return std::numeric_limits<scalar_t>::lowest();
    else
      return (scalar_t)0;
  }

  static inline __host__ __device__ void update(scalar_t *val,
                                                scalar_t new_val) {
    if (REDUCE == SUM || REDUCE == MEAN)
      *val = *val + new_val;
    else if (REDUCE == MUL)
      *val = *val * new_val;
    else if (REDUCE == DIV)
      *val = *val / new_val;
    else if ((REDUCE == MIN && new_val < *val) ||
             (REDUCE == MAX && new_val > *val)) {
      *val = new_val;
    }
  }

  static inline __host__ __device__ void update(scalar_t *val, scalar_t new_val,
                                                int64_t *arg, int64_t new_arg) {
    if (REDUCE == SUM || REDUCE == MEAN)
      *val = *val + new_val;
    else if (REDUCE == MUL)
      *val = *val * new_val;
    else if (REDUCE == DIV)
      *val = *val / new_val;
    else if ((REDUCE == MIN && new_val < *val) ||
             (REDUCE == MAX && new_val > *val)) {
      *val = new_val;
      *arg = new_arg;
    }
  }

  static inline __host__ __device__ void write(scalar_t *address, scalar_t val,
                                               int64_t *arg_address,
                                               int64_t arg, int count) {
    if (REDUCE == SUM || REDUCE == MUL || REDUCE == DIV)
      *address = val;
    else if (REDUCE == MEAN)
      *address = val / (scalar_t)(count > 0 ? count : 1);
    else if (REDUCE == MIN || REDUCE == MAX) {
      if (count > 0) {
        *address = val;
        *arg_address = arg;
      } else
        *address = (scalar_t)0;
    }
  }

  static inline __device__ void atomic_write(scalar_t *address, scalar_t val) {
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