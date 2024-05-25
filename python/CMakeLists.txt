include(AddMLIRPython)

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=minimal.")


################################################################################
# Sources
################################################################################

declare_mlir_python_sources(MinimalPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT MinimalPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE Minimal.td
  SOURCES
    minimal.py
  DIALECT_NAME minimal)

declare_mlir_python_extension(MinimalPythonSources.Extension
  MODULE_NAME _minimal
  ADD_TO_PARENT MinimalPythonSources
  SOURCES
    MinimalExtension.cpp
  EMBED_CAPI_LINK_LIBS
    MinimalCAPI
)

################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(MinimalPythonCAPI
  INSTALL_COMPONENT MinimalPythonModules
  INSTALL_DESTINATION python_packages/minimal/mlir_minimal/_mlir_libs
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/python_packages/minimal/mlir_minimal/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    MinimalPythonSources
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
)

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(MinimalPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/python_packages/minimal/mlir_minimal"
  INSTALL_PREFIX "python_packages/minimal/mlir_minimal"
  DECLARED_SOURCES
    MinimalPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources
  COMMON_CAPI_LINK_LIBS
    MinimalPythonCAPI
  )
