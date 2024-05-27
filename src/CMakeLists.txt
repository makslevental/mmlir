add_mlir_dialect_library(
  MLIRMinimal
  MinimalDialect.cpp

  # to silence LLVMProcessSources.cmake:116
  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include

  DEPENDS
  MLIRMinimalIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRFuncDialect
)

add_mlir_public_c_api_library(MinimalCAPI
  MinimalCAPI.cpp
  # to silence LLVMProcessSources.cmake:116
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRMinimal
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS ${dialect_libs} ${conversion_libs} MLIROptLib MLIRMinimal)

add_llvm_executable(minimal-opt minimal-opt.cpp)
llvm_update_compile_flags(minimal-opt)
target_link_libraries(minimal-opt PRIVATE ${LIBS})
add_dependencies(minimal-opt MLIRMinimalIncGen)

mlir_check_all_link_libraries(minimal-opt)