//===- MinimalCAPI.h - CAPI for minimal dialect -----------------------------*- C -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MINIMAL_C_DIALECTS_H
#define MINIMAL_C_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Minimal, minimal);

#ifdef __cplusplus
}
#endif

#endif // MINIMAL_C_DIALECTS_H
