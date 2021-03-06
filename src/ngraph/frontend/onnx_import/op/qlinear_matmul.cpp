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

#include "qlinear_matmul.hpp"
#include "frontend/onnx_import/utils/matmul_factory.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector qlinear_matmul(const Node& node)
                {
                    auto factory = matmul::QLinearMatmulFactory(node);
                    return factory.make_matmul_op();
                }
            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
