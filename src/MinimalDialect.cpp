//===- MinimalDialect.cpp - Minimal dialect ---------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MinimalDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::minimal;

#include "MinimalDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Minimal dialect.
//===----------------------------------------------------------------------===//

void MinimalDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "MinimalOps.cpp.inc"
      >();
  registerTypes();
}

//===----------------------------------------------------------------------===//
// Minimal ops
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "MinimalOps.cpp.inc"

namespace mlir::minimal {
#define GEN_PASS_DEF_MINIMALSWITCHBARFOO
#include "MinimalPasses.h.inc"

//===----------------------------------------------------------------------===//
// Minimal passes
//===----------------------------------------------------------------------===//

namespace {
class MinimalSwitchBarFooRewriter : public OpRewritePattern<func::FuncOp> {
public:
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter &rewriter) const final {
    if (op.getSymName() == "bar") {
      rewriter.modifyOpInPlace(op, [&op]() { op.setSymName("foo"); });
      return success();
    }
    return failure();
  }
};

class MinimalSwitchBarFoo
    : public impl::MinimalSwitchBarFooBase<MinimalSwitchBarFoo> {
public:
  using impl::MinimalSwitchBarFooBase<
      MinimalSwitchBarFoo>::MinimalSwitchBarFooBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<MinimalSwitchBarFooRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};
} // namespace
} // namespace mlir::minimal

//===----------------------------------------------------------------------===//
// Minimal types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "MinimalTypes.cpp.inc"

void MinimalDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "MinimalTypes.cpp.inc"
      >();
}
