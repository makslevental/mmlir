//===- Dialects.cpp - CAPI for dialects -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MinimalCAPI.h"

#include "MinimalDialect.h"

#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Minimal, minimal,
                                      mlir::minimal::MinimalDialect)

//===---------------------------------------------------------------------===//
// CustomType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAMinimalCustomType(MlirType type) {
  return llvm::isa<mlir::minimal::CustomType>(unwrap(type));
}

MlirType mlirMinimalCustomTypeGet(MlirContext ctx, MlirStringRef value) {
  return wrap(mlir::minimal::CustomType::get(unwrap(ctx), unwrap(value)));
}