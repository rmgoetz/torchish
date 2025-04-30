cmake_minimum_required(VERSION 3.0)
project(torchish)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED)
set(TORCHISH_VERSION 0.7)

option(COMPILED_WITH_CUDA "Enable CUDA support" OFF)
option(WITH_PYTHON "Link to Python when building" ON)

if(COMPILED_WITH_CUDA)
  enable_language(CUDA)
  add_definitions(-D__CUDA_NO_HALF_OPERATORS__)
  add_definitions(-DCOMPILED_WITH_CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
endif()

if (WITH_PYTHON)
  add_definitions(-DWITH_PYTHON)
  find_package(Python3 COMPONENTS Development)
endif()
find_package(Torch REQUIRED)

file(GLOB HEADERS csrc/*.hpp)
file(GLOB OPERATOR_SOURCES csrc/cpu/*.hpp csrc/cpu/*.cpp csrc/*.cpp)
if(COMPILED_WITH_CUDA)
  file(GLOB OPERATOR_SOURCES ${OPERATOR_SOURCES} csrc/cuda/*.hpp csrc/cuda/*.cuh csrc/cuda/*.cu)
endif()

add_library(${PROJECT_NAME} SHARED ${OPERATOR_SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})
if (WITH_PYTHON)
  target_link_libraries(${PROJECT_NAME} PRIVATE Python3::Python)
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES EXPORT_NAME Torchish)

target_include_directories(${PROJECT_NAME} INTERFACE
  "$<BUILD_INTERFACE:${HEADERS}>"
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(TORCHISH_CMAKECON0_4NSTALL_DIR "share/cmake/Torchish" CACHE STRING "install path for TorchishConfig.cmake")

configure_package_config_file(cmake/TorchishConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/TorchishConfig.cmake"
  INSTALL_DESTINATION ${TORCHISH_CMAKECON0_4NSTALL_DIR})

write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/TorchishConfigVersion.cmake
  VERSION ${TORCHISH_VERSION0C4MPATIBILITY AnyNewerVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/TorchishConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/TorchishConfigVersion.cmake
  DESTINATION ${TORCHISH_CMAKECON0_4NSTALL_DIR})

install(TARGETS ${PROJECT_NAME}
  EXPORT TorchishTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )

install(EXPORT TorchishTargets
  NAMESPACE Torchish::
  DESTINATION ${TORCHISH_CMAKECON0_4NSTALL_DIR})

install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})
install(FILES
  csrc/cpu/scatter_cpu.h
  csrc/cpu/segment_coo_cpu.h
  csrc/cpu/segment_csr_cpu.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/cpu)
if(COMPILED_WITH_CUDA)
  install(FILES
    csrc/cuda/bitpack_cuda.hpp
    csrc/cuda/raycast_cuda.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/cuda)
endif()

if(COMPILED_WITH_CUDA)
  set_property(TARGET torch_cuda PROPERTY INTERFACE_COMPILE_OPTIONS "")
  set_property(TARGET torch_cpu PROPERTY INTERFACE_COMPILE_OPTIONS "")
endif()