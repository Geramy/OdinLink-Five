# Host LZ4 only (ODLC algo 4). nvCOMP is not wired: it does not shrink
# NCCL DMA (fixed max_wire), a Mac cannot decode GDeflate, and GeForce
# Blackwell has no decompress engine. odl_nvcomp.cpp stays in the tree
# unused.

message(STATUS "odl_compress: host LZ4 (nvCOMP not linked)")

# Host LZ4 (algo 4) — Mac-readable format.
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

target_sources(odl_compress PRIVATE src/odl_compress_stub.c)

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

# Host LZ4 round-trip — no CUDA, runs on Linux and Mac CI.
add_executable(odl_compress_host_test tests/odl_compress_host_test.c)
target_link_libraries(odl_compress_host_test PRIVATE odl_compress)
target_include_directories(odl_compress_host_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
