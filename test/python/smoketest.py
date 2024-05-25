# RUN: %python %s | FileCheck %s

from mmlir.ir import *
from mmlir.dialects import minimal as minimal_d

with Context():
    minimal_d.register_dialect()
    module = Module.parse(
        """
    %0 = arith.constant 2 : i32
    %1 = minimal.foo %0 : i32
    """
    )
    # CHECK: %[[C:.*]] = arith.constant 2 : i32
    # CHECK: minimal.foo %[[C]] : i32
    print(str(module))
