// RUN: minimal-opt %s | minimal-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        %0 = arith.constant 1 : i32
        // CHECK: %{{.*}} = minimal.foo %{{.*}} : i32
        %res = minimal.foo %0 : i32
        return
    }

    // CHECK-LABEL: func @minimal_types(%arg0: !minimal.custom<"10">)
    func.func @minimal_types(%arg0: !minimal.custom<"10">) {
        return
    }
}
