llvm_canonicalize_cmake_booleans(MLIR_ENABLE_BINDINGS_PYTHON)

# this configures the adjacent lit.site.cfg.py.in
set(MINIMAL_BINARY_DIR ${PROJECT_BINARY_DIR})
set(MINIMAL_SOURCE_DIR ${PROJECT_SOURCE_DIR})
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

set(MINIMAL_TEST_DEPENDS
    FileCheck
    count
    not
    minimal-opt)

if(MLIR_ENABLE_BINDINGS_PYTHON)
  list(APPEND MINIMAL_TEST_DEPENDS MinimalPythonModules)
endif()

add_lit_testsuite(
  check-minimal "Running the minimal regression tests"
  ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${MINIMAL_TEST_DEPENDS})
set_target_properties(check-minimal PROPERTIES FOLDER "Tests")

add_lit_testsuites(MINIMAL ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS
                   ${MINIMAL_TEST_DEPENDS})
