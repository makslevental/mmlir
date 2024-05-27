//===- MinimalDialect.h - Minimal dialect -----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MINIMAL_MINIMALDIALECT_H
#define MINIMAL_MINIMALDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include <memory>

#include "MinimalDialect.h.inc"

#define GET_OP_CLASSES
#include "MinimalOps.h.inc"

namespace mlir::minimal {
#define GEN_PASS_DECL
#include "MinimalPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "MinimalPasses.h.inc"
} // namespace mlir::minimal

#define GET_TYPEDEF_CLASSES
#include "MinimalTypes.h.inc"

#endif // MINIMAL_MINIMALDIALECT_H
