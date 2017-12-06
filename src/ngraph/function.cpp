// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include <memory>

#include "ngraph/function.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

atomic<size_t> Function::m_next_instance_id(0);

Function::Function(const std::shared_ptr<Node>& result,
                   const std::shared_ptr<const ValueType>& result_type,
                   const std::vector<std::shared_ptr<op::Parameter>>& parameters,
                   const std::string& name)
    : m_result(result)
    , m_parameters(parameters)
    , m_name(name)
    , m_ordered_ops_valid(false)
    , m_temporary_pool_size(0)
    , m_instance_id(m_next_instance_id.fetch_add(1))
{
    if (nullptr == result->get_value_type())
    {
        throw ngraph_error("Internal nGraph error: result->get_value_type() == nullptr");
    }

    if (nullptr != result_type && (*result_type != *(result->get_value_type())))
    {
        throw ngraph_error("Function result node's value type does not match declared return type");
    }

    traverse_nodes(this, [&](shared_ptr<Node> node) {
        m_ops.push_back(node);

        std::shared_ptr<op::Parameter> p = std::dynamic_pointer_cast<op::Parameter>(node);
        if (nullptr != p)
        {
            auto it = std::find_if(parameters.begin(),
                                   parameters.end(),
                                   [p](std::shared_ptr<op::Parameter> q) { return (p == q); });
            if (it == parameters.end())
            {
                throw ngraph_error("Function references undeclared parameter");
            }
        }
    });
}

Function::Function(const std::shared_ptr<Node>& result,
                   const std::vector<std::shared_ptr<op::Parameter>>& parameters,
                   const std::string& name)
    : Function(result, nullptr, parameters, name)
{
}

void Function::set_ordered_ops(const std::list<shared_ptr<Node>>& ordered_ops)
{
    m_ordered_ops = ordered_ops;
    m_ordered_ops_valid = true;
}

std::list<shared_ptr<Node>>& Function::get_ops()
{
    return m_ops;
}

const std::list<shared_ptr<Node>>& Function::get_ops() const
{
    return m_ops;
}

std::list<shared_ptr<Node>>& Function::get_ordered_ops()
{
    if (!m_ordered_ops_valid)
    {
        throw ngraph_error("Access to ordered ops invalid");
    }
    return m_ordered_ops;
}

const std::list<shared_ptr<Node>>& Function::get_ordered_ops() const
{
    if (!m_ordered_ops_valid)
    {
        throw ngraph_error("Access to ordered ops invalid");
    }
    return m_ordered_ops;
}

std::string Function::get_name() const
{
    string rc;
    if (m_name.empty())
    {
        rc = "Function_" + to_string(m_instance_id);
    }
    else
    {
        rc = m_name;
    }
    return rc;
}

void Function::set_name(const string& name)
{
    if (m_name.empty())
    {
        m_name = name;
    }
    else
    {
        throw ngraph_error("Function name may be set exactly once");
    }
}

size_t Function::get_temporary_pool_size()
{
    return m_temporary_pool_size;
}

void Function::set_temporary_pool_size(size_t size)
{
    m_temporary_pool_size = size;
}

std::ostream& ngraph::operator<<(std::ostream& out, const Function& f)
{
    out << "Function(" << f.get_name() << ")";
    return out;
}
