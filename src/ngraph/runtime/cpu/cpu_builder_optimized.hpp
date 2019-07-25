//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <string>
#include <vector>

#include "ngraph/node.hpp"
#include "ngraph/runtime/cpu/cpu_external_function.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view_wrapper.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"

#define SELECT_KERNEL_DOT(KV, ET, R1, R2, R3, K)                                                   \
    if (ET == element::f32)                                                                        \
    {                                                                                              \
        KV = K<float, R1, R2, R3>;                                                                 \
    }                                                                                              \
    else if (ET == element::i64)                                                                   \
    {                                                                                              \
        KV = K<int64_t, R1, R2, R3>;                                                               \
    }                                                                                              \
    else                                                                                           \
    {                                                                                              \
        throw ngraph_error("Unsupported element type " + ET.c_type_string() + " for kernel " #K);  \
    }

#define SELECT_KERNEL_SOFTMAX(KV, ET, K)                                                           \
    if (ET == element::f32)                                                                        \
    {                                                                                              \
        KV = K<float>;                                                                             \
    }                                                                                              \
    else if (ET == element::i64)                                                                   \
    {                                                                                              \
        KV = K<int64_t>;                                                                           \
    }                                                                                              \
    else                                                                                           \
    {                                                                                              \
        throw ngraph_error("Unsupported element type " + ET.c_type_string() + " for kernel " #K);  \
    }

 #define PARTIAL_SELECT_SOFTMAX_BY_RANK(KV, ET, R, K)                                               \
    if (ET == element::f32)                                                                        \
    {                                                                                              \
        PARTIAL_SELECT_RANK(KV, float, R, K);                                                      \
    }                                                                                              \
    else if (ET == element::i64)                                                                   \
    {                                                                                              \
        PARTIAL_SELECT_RANK(KV, int64_t, R, K);                                                    \
    }                                                                                              \
    else                                                                                           \
    {                                                                                              \
        throw ngraph_error("Unsupported element type " + ET.c_type_string() + " for kernel " #K);  \
    }

#define SELECT_REDUCTION_BY_RANK(KV, ET, R, K)                                                     \
    if (ET == element::f32)                                                                        \
    {                                                                                              \
        SELECT_RANK(KV, float, R, K);                                                              \
    }                                                                                              \
    else if (ET == element::i64)                                                                   \
    {                                                                                              \
        SELECT_RANK(KV, int64_t, R, K);                                                            \
    }                                                                                              \
    else                                                                                           \
    {                                                                                              \
        throw ngraph_error("Unsupported element type " + ET.c_type_string() + " for kernel " #K);  \
    }

#define SELECT_REDUCTION(KV, ET, K)                                                                \
    if (ET == element::f32)                                                                        \
    {                                                                                              \
        KV = K<float>;                                                                             \
    }                                                                                              \
    else if (ET == element::i64)                                                                   \
    {                                                                                              \
        KV = K<int64_t>;                                                                           \
    }                                                                                              \
    else                                                                                           \
    {                                                                                              \
        throw ngraph_error("Unsupported element type " + ET.c_type_string() + " for kernel " #K);  \
    }

