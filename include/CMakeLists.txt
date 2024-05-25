# this is so tablegen emits stuff into this dir
set(CMAKE_CURRENT_BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(dialect Minimal)
set(dialect_namespace minimal)
set(LLVM_TARGET_DEFINITIONS ${dialect}Dialect.td)

mlir_tablegen(${dialect}Ops.h.inc -gen-op-decls)
mlir_tablegen(${dialect}Ops.cpp.inc -gen-op-defs)
mlir_tablegen(${dialect}Types.h.inc -gen-typedef-decls
              -typedefs-dialect=${dialect_namespace})
mlir_tablegen(${dialect}Types.cpp.inc -gen-typedef-defs
              -typedefs-dialect=${dialect_namespace})
mlir_tablegen(${dialect}Dialect.h.inc -gen-dialect-decls
              -dialect=${dialect_namespace})
mlir_tablegen(${dialect}Dialect.cpp.inc -gen-dialect-defs
              -dialect=${dialect_namespace})
mlir_tablegen(MinimalPasses.h.inc --gen-pass-decls)

add_public_tablegen_target(MLIR${dialect}IncGen)
