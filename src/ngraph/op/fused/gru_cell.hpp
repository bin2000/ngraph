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

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "ngraph/node.hpp"
#include "ngraph/op/util/activation_functions.hpp"
#include "ngraph/op/util/fused_op.hpp"
#include "ngraph/op/util/rnn_cell_base.hpp"

namespace ngraph
{
    namespace op
    {
        ///
        /// \brief      Class for GRU cell node.
        ///
        /// \note       It follows notation and equations defined as in ONNX standard:
        ///             https://github.com/onnx/onnx/blob/master/docs/Operators.md#GRU
        ///
        ///             Note this class represents only single *cell* and not whole GRU *layer*.
        ///
        class GRUCell : public util::FusedOp, public util::RNNCellBase
        {
        public:
            ///
            /// \brief      Constructs GRUCell node.
            ///
            /// \param[in]  X                 The input tensor with shape: [batch_size, input_size].
            /// \param[in]  W                 The weight tensor with shape:
            ///                               [gates_count * hidden_size, input_size].
            /// \param[in]  R                 The recurrence weight tensor with shape:
            ///                               [gates_count * hidden_size, hidden_size].
            /// \param[in]  H_t               The hidden state tensor at current time step with
            ///                               shape: [batch_size, hidden_size].
            /// \param[in]  hidden_size  The number of hidden units for recurrent cell.
            ///
            GRUCell(const std::shared_ptr<Node>& X,
                    const std::shared_ptr<Node>& W,
                    const std::shared_ptr<Node>& R,
                    const std::shared_ptr<Node>& H_t,
                    std::size_t hidden_size);

            ///
            /// \brief      Constructs GRUCell node.
            ///
            /// \param[in]  X                 The input tensor with shape: [batch_size, input_size].
            /// \param[in]  W                 The weight tensor with shape:
            ///                               [gates_count * hidden_size, input_size].
            /// \param[in]  R                 The recurrence weight tensor with shape:
            ///                               [gates_count * hidden_size, hidden_size].
            /// \param[in]  H_t               The hidden state tensor at current time step with
            ///                               shape: [batch_size, hidden_size].
            /// \param[in]  hidden_size       The number of hidden units for recurrent cell.
            /// \param[in]  activations       The vector of activation functions used inside
            ///                               recurrent cell.
            /// \param[in]  activation_alpha  The vector of alpha parameters for activation
            ///                               functions in order respective to activation list.
            /// \param[in]  activation_beta   The vector of beta parameters for activation functions
            ///                               in order respective to activation list.
            /// \param[in]  clip              The value defining clipping range [-clip, clip] on
            ///                               input of activation functions.
            ///
            GRUCell(const std::shared_ptr<Node>& X,
                    const std::shared_ptr<Node>& W,
                    const std::shared_ptr<Node>& R,
                    const std::shared_ptr<Node>& H_t,
                    std::size_t hidden_size,
                    const std::vector<std::string>& activations,
                    const std::vector<float>& activation_alpha,
                    const std::vector<float>& activation_beta,
                    float clip,
                    bool linear_before_reset);

            ///
            /// \brief      Constructs GRUCell node.
            ///
            /// \param[in]  X                 The input tensor with shape: [batch_size, input_size].
            /// \param[in]  W                 The weight tensor with shape:
            ///                               [gates_count * hidden_size, input_size].
            /// \param[in]  R                 The recurrence weight tensor with shape:
            ///                               [gates_count * hidden_size, hidden_size].
            /// \param[in]  H_t               The hidden state tensor at current time step with
            ///                               shape: [batch_size, hidden_size].
            /// \param[in]  hidden_size       The number of hidden units for recurrent cell.
            /// \param[in]  B                 The bias tensor for input gate with shape:
            ///                               [2 * gates_count * hidden_size].
            /// \param[in]  activations       The vector of activation functions used inside
            ///                               recurrent cell.
            /// \param[in]  activation_alpha  The vector of alpha parameters for activation
            ///                               functions in order respective to activation list.
            /// \param[in]  activation_beta   The vector of beta parameters for activation functions
            ///                               in order respective to activation list.
            /// \param[in]  clip              The value defining clipping range [-clip, clip] on
            ///                               input of activation functions.
            ///
            GRUCell(const std::shared_ptr<Node>& X,
                    const std::shared_ptr<Node>& W,
                    const std::shared_ptr<Node>& R,
                    const std::shared_ptr<Node>& H_t,
                    std::size_t hidden_size,
                    const std::shared_ptr<Node>& B,
                    const std::vector<std::string>& activations =
                        std::vector<std::string>{"sigmoid", "tanh"},
                    const std::vector<float>& activation_alpha = {},
                    const std::vector<float>& activation_beta = {},
                    float clip = 0.f,
                    bool linear_before_reset = false);

            virtual void pre_validate_and_infer_types() override;
            virtual NodeVector decompose_op() const override;
            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            bool get_linear_before_reset() const { return m_linear_before_reset; }
        private:
            /// brief Add and initialize bias input to all zeros.
            void add_default_bias_input();

            ///
            /// \brief The Activation function f.
            ///
            util::ActivationFunction m_activation_f;
            ///
            /// \brief The Activation function g.
            ///
            util::ActivationFunction m_activation_g;

            static constexpr std::size_t s_gates_count{3};
            ///
            /// \brief Control whether or not apply the linear transformation.
            ///
            /// \note The linear transformation may be applied when computing the output of hidden gate.
            ///       It's done before multiplying by the output of the reset gate.
            ///
            bool m_linear_before_reset;
        };
    }
}
