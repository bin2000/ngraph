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

#include "type.hpp"

#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/STLExtras.h"
#include "ngraph/assertion.hpp"

using llvm::ArrayRef;
using llvm::raw_ostream;
using llvm::raw_string_ostream;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

namespace ngraph
{
    using namespace runtime::cpu;
    /// Creates TensorType objects. They all point to the same storage if
    /// element type and shape are the same.
    NGTensorType NGTensorType::get(mlir::MLIRContext* context, EltType eltType, Shape shape)
    {
        return Base::get(context, NGTypeKind::TENSOR_TYPE_ID, eltType, shape);
    }

    mlir::MemRefType NGTensorType::toMemref()
    {
        auto memRefType =
            mlir::MemRefType::get(getShape(), getElementType(), {/* no map used */}, 0);
        return memRefType;
    }
}