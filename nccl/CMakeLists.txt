add_library(nccl-net-ODL_TB5 SHARED
    src/odl_tb5_nccl_plugin.c
)

target_include_directories(nccl-net-ODL_TB5 PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/nccl
    ${CMAKE_SOURCE_DIR}/lib/include
)

find_package(Threads REQUIRED)
target_link_libraries(nccl-net-ODL_TB5
    odl_tb5
    Threads::Threads
)

# For dlopen/libdl compatibility
if(CMAKE_DL_LIBS)
    target_link_libraries(nccl-net-ODL_TB5 ${CMAKE_DL_LIBS})
endif()

set_target_properties(nccl-net-ODL_TB5 PROPERTIES
    LIBRARY_OUTPUT_NAME "nccl-net-ODL_TB5"
    SOVERSION 0
)

include(GNUInstallDirs)
install(TARGETS nccl-net-ODL_TB5
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT nccl
)
