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
# RCCL probes librccl-net.so (exact name) via LD_LIBRARY_PATH / plugin dir.
# Install a second name so users do not need a manual symlink (issue #23).
install(FILES $<TARGET_FILE:rccl_net_odl_tb5>
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RENAME librccl-net.so
    COMPONENT rccl
)
# Convenience copy next to the build tree artifact for local (non-install) runs.
add_custom_command(TARGET rccl_net_odl_tb5 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        $<TARGET_FILE_NAME:rccl_net_odl_tb5>
        $<TARGET_FILE_DIR:rccl_net_odl_tb5>/librccl-net.so
    COMMENT "Symlink librccl-net.so → librccl_net_odl_tb5.so for RCCL probe"
)
