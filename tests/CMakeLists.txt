add_executable(odl_tb5_test
    odl_tb5_test_main.c
    odl_tb5_test_device.c
    odl_tb5_test_lib_api.c
    odl_tb5_test_plugin.c
)

target_include_directories(odl_tb5_test PRIVATE
    ${CMAKE_SOURCE_DIR}/lib/include
    ${CMAKE_SOURCE_DIR}/driver/uapi
    ${CMAKE_SOURCE_DIR}/third_party/rccl
)
target_link_libraries(odl_tb5_test odl_tb5 rccl_net_odl_tb5)
if(NOT APPLE)
    target_link_libraries(odl_tb5_test dl)
endif()

add_executable(odl_tb5_test_rccl_dmabuf
    odl_tb5_test_rccl_dmabuf.c
)

target_include_directories(odl_tb5_test_rccl_dmabuf PRIVATE
    ${CMAKE_SOURCE_DIR}/lib/include
    ${CMAKE_SOURCE_DIR}/driver/uapi
)
target_link_libraries(odl_tb5_test_rccl_dmabuf odl_tb5 pthread)
