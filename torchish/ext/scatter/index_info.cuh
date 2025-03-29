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
 *  furnished to do so, subject to the following conditions:
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
 *  THE SOFTWARE.
 * 
 */

#pragma once

#include <ATen/cuda/detail/TensorInfo.cuh>

// We need our own `IndexToOffset` implementation since we do not want to
// access the last element of the `indexptr`.
template <typename scalar_t> struct IndexPtrToOffset {
  static inline __host__ __device__ int
  get(int idx, const at::cuda::detail::TensorInfo<scalar_t, int> &info) {
    int offset = idx % (info.sizes[info.dims - 1] - 1);
    offset *= info.strides[info.dims - 1];
    idx /= info.sizes[info.dims - 1] - 1;
    for (int i = info.dims - 2; i >= 0; --i) {
      offset += (idx % info.sizes[i]) * info.strides[i];
      idx /= info.sizes[i];
    }
    return offset;
  }
};