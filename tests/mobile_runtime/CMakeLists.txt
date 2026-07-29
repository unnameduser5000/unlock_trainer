cmake_minimum_required(VERSION 3.19)
project(sid_mobile_runtime_harness LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT EXECUTORCH_ROOT)
  message(FATAL_ERROR "Set EXECUTORCH_ROOT to the patched ExecuTorch source directory")
endif()

find_package(executorch CONFIG REQUIRED)
target_include_directories(executorch INTERFACE "${EXECUTORCH_ROOT}/..")

add_executable(bpfree_module_harness bpfree_module_harness.cpp)
target_include_directories(bpfree_module_harness PRIVATE "${EXECUTORCH_ROOT}/..")
target_link_libraries(
  bpfree_module_harness
  PRIVATE
    executorch
    portable_ops_lib
    portable_kernels
    extension_data_loader
    extension_tensor
    extension_training
)
target_compile_options(bpfree_module_harness PRIVATE -Wno-deprecated-declarations)
