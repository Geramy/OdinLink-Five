add_library(rccl_net_tb5 SHARED src/tb5_plugin.c)
target_include_directories(rccl_net_tb5 PRIVATE ${CMAKE_SOURCE_DIR}/third_party/rccl)
target_link_libraries(rccl_net_tb5 tb5_ring)
