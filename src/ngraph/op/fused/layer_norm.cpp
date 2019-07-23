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
#include "ngraph/op/fused/layer_norm.hpp"

#include "ngraph/builder/make_constant.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/util/broadcasting.hpp"

using namespace std;
using namespace ngraph;

op::LayerNorm::LayerNorm(const shared_ptr<Node>& data)
    : FusedOp("LayerNorm", {data})
{
    constructor_validate_and_infer_types();
}

NodeVector op::LayerNorm::decompose_op() const
{
    auto data = get_argument(0);

    return {data};
}

shared_ptr<Node> op::LayerNorm::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<LayerNorm>(new_args.at(0));
}
