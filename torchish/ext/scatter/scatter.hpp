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

#include "extensions.hpp"

namespace scatter {
SCATTER_API int64_t cuda_version() noexcept;

namespace detail {
SCATTER_INLINE_VARIABLE int64_t _cuda_version = cuda_version();
} // namespace detail
} // namespace scatter

// SCATTER_API torch::Tensor
// scatter_sum(torch::Tensor src, torch::Tensor index, int64_t dim,
//             std::optional<torch::Tensor> optional_out,
//             std::optional<int64_t> dim_size);

// SCATTER_API torch::Tensor
// scatter_mul(torch::Tensor src, torch::Tensor index, int64_t dim,
//             std::optional<torch::Tensor> optional_out,
//             std::optional<int64_t> dim_size);

// SCATTER_API torch::Tensor
// scatter_mean(torch::Tensor src, torch::Tensor index, int64_t dim,
//              std::optional<torch::Tensor> optional_out,
//              std::optional<int64_t> dim_size);

SCATTER_API std::tuple<torch::Tensor, torch::Tensor>
scatter_min(torch::Tensor src, torch::Tensor index, int64_t dim,
            std::optional<torch::Tensor> optional_out,
            std::optional<int64_t> dim_size);

SCATTER_API std::tuple<torch::Tensor, torch::Tensor>
scatter_max(torch::Tensor src, torch::Tensor index, int64_t dim,
            std::optional<torch::Tensor> optional_out,
            std::optional<int64_t> dim_size);

// SCATTER_API torch::Tensor
// segment_sum_coo(torch::Tensor src, torch::Tensor index,
//                 std::optional<torch::Tensor> optional_out,
//                 std::optional<int64_t> dim_size);

// SCATTER_API torch::Tensor
// segment_mean_coo(torch::Tensor src, torch::Tensor index,
//                  std::optional<torch::Tensor> optional_out,
//                  std::optional<int64_t> dim_size);

// SCATTER_API std::tuple<torch::Tensor, torch::Tensor>
// segment_min_coo(torch::Tensor src, torch::Tensor index,
//                 std::optional<torch::Tensor> optional_out,
//                 std::optional<int64_t> dim_size);

// SCATTER_API std::tuple<torch::Tensor, torch::Tensor>
// segment_max_coo(torch::Tensor src, torch::Tensor index,
//                 std::optional<torch::Tensor> optional_out,
//                 std::optional<int64_t> dim_size);

// SCATTER_API torch::Tensor
// gather_coo(torch::Tensor src, torch::Tensor index,
//            std::optional<torch::Tensor> optional_out);

// SCATTER_API torch::Tensor
// segment_sum_csr(torch::Tensor src, torch::Tensor indptr,
//                 std::optional<torch::Tensor> optional_out);

// SCATTER_API torch::Tensor
// segment_mean_csr(torch::Tensor src, torch::Tensor indptr,
//                  std::optional<torch::Tensor> optional_out);

// SCATTER_API std::tuple<torch::Tensor, torch::Tensor>
// segment_min_csr(torch::Tensor src, torch::Tensor indptr,
//                 std::optional<torch::Tensor> optional_out);

// SCATTER_API std::tuple<torch::Tensor, torch::Tensor>
// segment_max_csr(torch::Tensor src, torch::Tensor indptr,
//                 std::optional<torch::Tensor> optional_out);

// SCATTER_API torch::Tensor
// gather_csr(torch::Tensor src, torch::Tensor indptr,
//            std::optional<torch::Tensor> optional_out);