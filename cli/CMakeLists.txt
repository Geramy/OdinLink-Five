add_executable(odl_tb5_cli
    src/odl_tb5_cli_main.c
    src/odl_tb5_cli_server.c
    src/odl_tb5_cli_client.c
    src/odl_tb5_cli_bandwidth.c
    src/odl_tb5_cli_latency.c
    src/odl_tb5_cli_latency_load.c
    src/odl_tb5_cli_mimo.c
    src/odl_tb5_cli_jitter.c
    src/odl_tb5_cli_proto.c
    src/odl_tb5_cli_stats.c
    src/odl_tb5_cli_diag.c
)

target_include_directories(odl_tb5_cli PRIVATE
    src
    ${CMAKE_SOURCE_DIR}/lib/include
    ${CMAKE_SOURCE_DIR}/driver/uapi
)
target_link_libraries(odl_tb5_cli odl_tb5 pthread m)

include(GNUInstallDirs)
install(TARGETS odl_tb5_cli
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT cli
)
