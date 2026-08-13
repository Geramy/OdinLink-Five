# Optional GPU compression (nvCOMP). Never fails the rest of the tree.
#
#   cmake -DODL_ENABLE_NVCOMP=ON|OFF|AUTO   (default AUTO)
#   -DNVCOMP_ROOT=/path/to/nvcomp           (optional hint)
#
# Without nvCOMP, a stub backend is linked: odl_compress_enabled() is always 0
# and NCCL keeps the uncompressed path.

option(ODL_ENABLE_NVCOMP "Build nvCOMP compression backend (ON/OFF/AUTO)" AUTO)

set(ODL_NVCOMP_FOUND FALSE)
set(ODL_NVCOMP_INCLUDE_DIR "")
set(ODL_NVCOMP_LIBRARY "")

if(NOT ODL_ENABLE_NVCOMP STREQUAL "OFF")
  # Search order: explicit root, system prefix, pip/conda nvidia wheel layout
  set(_nvcomp_hints "")
  if(NVCOMP_ROOT)
    list(APPEND _nvcomp_hints "${NVCOMP_ROOT}")
  endif()
  if(DEFINED ENV{NVCOMP_ROOT})
    list(APPEND _nvcomp_hints "$ENV{NVCOMP_ROOT}")
  endif()
  list(APPEND _nvcomp_hints
    /usr
    /usr/local
    /usr/local/cuda
  )
  if(DEFINED ENV{CONDA_PREFIX})
    list(APPEND _nvcomp_hints
      "$ENV{CONDA_PREFIX}"
      "$ENV{CONDA_PREFIX}/lib/python3.13/site-packages/nvidia/libnvcomp"
      "$ENV{CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/libnvcomp"
      "$ENV{CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/libnvcomp"
    )
  endif()
  if(DEFINED ENV{HOME})
    list(APPEND _nvcomp_hints
      "$ENV{HOME}/miniconda3/lib/python3.13/site-packages/nvidia/libnvcomp"
      "$ENV{HOME}/miniconda3/lib/python3.12/site-packages/nvidia/libnvcomp"
    )
  endif()

  # Manual scan — more reliable than find_* for pip wheels that lack unversioned .so
  foreach(_root ${_nvcomp_hints})
    if(NOT ODL_NVCOMP_INCLUDE_DIR AND EXISTS "${_root}/include/nvcomp/nvcompManager.hpp")
      set(ODL_NVCOMP_INCLUDE_DIR "${_root}/include")
    endif()
    if(NOT ODL_NVCOMP_INCLUDE_DIR AND EXISTS "${_root}/nvcomp/nvcompManager.hpp")
      set(ODL_NVCOMP_INCLUDE_DIR "${_root}")
    endif()
    if(NOT ODL_NVCOMP_LIBRARY AND EXISTS "${_root}/lib64/libnvcomp.so.5")
      set(ODL_NVCOMP_LIBRARY "${_root}/lib64/libnvcomp.so.5")
    endif()
    if(NOT ODL_NVCOMP_LIBRARY AND EXISTS "${_root}/lib/libnvcomp.so.5")
      set(ODL_NVCOMP_LIBRARY "${_root}/lib/libnvcomp.so.5")
    endif()
    if(NOT ODL_NVCOMP_LIBRARY AND EXISTS "${_root}/lib64/libnvcomp.so")
      set(ODL_NVCOMP_LIBRARY "${_root}/lib64/libnvcomp.so")
    endif()
  endforeach()

  find_package(CUDAToolkit QUIET)
  if(ODL_NVCOMP_INCLUDE_DIR AND ODL_NVCOMP_LIBRARY AND CUDAToolkit_FOUND)
    set(ODL_NVCOMP_FOUND TRUE)
  else()
    message(STATUS
      "nvCOMP probe: include=${ODL_NVCOMP_INCLUDE_DIR} lib=${ODL_NVCOMP_LIBRARY} "
      "cuda=${CUDAToolkit_FOUND}")
  endif()

  if(ODL_ENABLE_NVCOMP STREQUAL "ON" AND NOT ODL_NVCOMP_FOUND)
    message(WARNING
      "ODL_ENABLE_NVCOMP=ON but nvCOMP/CUDAToolkit not found.\n"
      "  Install (optional):\n"
      "    wget https://developer.download.nvidia.com/compute/nvcomp/5.3.0/local_installers/"
      "nvcomp-local-repo-ubuntu2604-5.3.0_5.3.0-1_amd64.deb\n"
      "    sudo dpkg -i nvcomp-local-repo-ubuntu2604-5.3.0_5.3.0-1_amd64.deb\n"
      "    sudo cp /var/nvcomp-local-repo-ubuntu2604-5.3.0/nvcomp-*-keyring.gpg /usr/share/keyrings/\n"
      "    sudo apt-get update && sudo apt-get -y install nvcomp\n"
      "  Or: pip install nvidia-nvcomp-cu12  and set\n"
      "    -DNVCOMP_ROOT=$CONDA_PREFIX/lib/python3.x/site-packages/nvidia/libnvcomp\n"
      "  Falling back to stub (compression disabled at runtime).")
  elseif(ODL_NVCOMP_FOUND)
    message(STATUS "nvCOMP found: ${ODL_NVCOMP_LIBRARY}")
    message(STATUS "nvCOMP include: ${ODL_NVCOMP_INCLUDE_DIR}")
  else()
    message(STATUS
      "nvCOMP not found — building compress stub (ODL_COMPRESS no-ops). "
      "Optional install: see docs/GPU.md §Compression")
  endif()
endif()

# Always produce libodl_compress so NCCL can link unconditionally.
# Host LZ4 (algo 4) is always compiled — that is the Mac-readable format.
add_library(odl_compress SHARED
  src/odl_compress_common.c
  src/odl_compress_lz4.c
  third_party/lz4/lz4.c
)

target_include_directories(odl_compress
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/lz4
)

if(APPLE)
  target_link_libraries(odl_compress PRIVATE "-framework Compression")
endif()

target_include_directories(odl_compress
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(ODL_NVCOMP_FOUND)
  enable_language(CXX)
  target_sources(odl_compress PRIVATE src/odl_nvcomp.cpp)
  target_compile_definitions(odl_compress PRIVATE ODL_HAS_NVCOMP=1)
  target_include_directories(odl_compress PRIVATE
    ${ODL_NVCOMP_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
  )
  target_link_libraries(odl_compress PRIVATE
    ${ODL_NVCOMP_LIBRARY}
    CUDA::cudart
  )
  # C++ ABI
  set_target_properties(odl_compress PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
  )
else()
  target_sources(odl_compress PRIVATE src/odl_compress_stub.c)
endif()

set_target_properties(odl_compress PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 0
  OUTPUT_NAME odl_compress
)

include(GNUInstallDirs)
install(TARGETS odl_compress
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  COMPONENT compress
)
install(DIRECTORY include/odl_tb5
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT compress
)

# Optional micro-bench (only with real nvCOMP)
if(ODL_NVCOMP_FOUND)
  add_executable(odl_compress_bench tests/odl_compress_bench.cpp)
  target_link_libraries(odl_compress_bench PRIVATE odl_compress CUDA::cudart)
  target_include_directories(odl_compress_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()

# Host LZ4 round-trip — no CUDA, runs on Linux and Mac CI.
add_executable(odl_compress_host_test tests/odl_compress_host_test.c)
target_link_libraries(odl_compress_host_test PRIVATE odl_compress)
target_include_directories(odl_compress_host_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
