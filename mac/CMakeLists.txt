# Linux-side sender for the Mac sink. The kext itself is built on macOS
# with mac/kext/Makefile.
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    return()
endif()

add_executable(odl_tensor_send linux_test/odl_tensor_send.c)
target_include_directories(odl_tensor_send PRIVATE
    ${CMAKE_SOURCE_DIR}/lib/include
    ${CMAKE_SOURCE_DIR}/mac/include
)
target_link_libraries(odl_tensor_send odl_tb5)

include(GNUInstallDirs)
install(TARGETS odl_tensor_send
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT cli
)
