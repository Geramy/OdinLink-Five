add_library(rccl_net_odl_tb5 SHARED
    src/odl_tb5_plugin.c
)

target_include_directories(rccl_net_odl_tb5 PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/rccl
    ${CMAKE_SOURCE_DIR}/lib/include
)
target_link_libraries(rccl_net_odl_tb5 odl_tb5)

include(GNUInstallDirs)
install(TARGETS rccl_net_odl_tb5
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT rccl
)
