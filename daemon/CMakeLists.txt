find_package(PkgConfig REQUIRED)
pkg_check_modules(GLIB2 REQUIRED glib-2.0>=2.56)
pkg_check_modules(GIO2  REQUIRED gio-2.0>=2.56)
pkg_check_modules(GIOUNIX REQUIRED gio-unix-2.0>=2.56)

# Optional: FUSE3 for transparent remote file access
pkg_check_modules(FUSE3 QUIET fuse3>=3.1)
if(FUSE3_FOUND)
    message(STATUS "FUSE3 found -- distributed file access enabled")
else()
    message(STATUS "FUSE3 not found -- distributed file access disabled. "
                   "Install libfuse3-dev to enable.")
endif()

# Optional: OpenSSL for SHA-256 (used by sync protocol)
pkg_check_modules(OPENSSL QUIET libcrypto)
if(OPENSSL_FOUND)
    message(STATUS "OpenSSL found -- SHA-256 enabled for file sync")
else()
    message(STATUS "OpenSSL not found -- file sync SHA-256 disabled. "
                   "Install libssl-dev to enable.")
endif()

# Generate D-Bus C bindings from introspection XML
set(DBUS_XML ${CMAKE_CURRENT_SOURCE_DIR}/dbus/com.odinlink.Tb5Daemon.xml)
set(DBUS_GEN_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/odl_tb5_dbus_generated)

add_custom_command(
    OUTPUT ${DBUS_GEN_PREFIX}.c ${DBUS_GEN_PREFIX}.h
    COMMAND gdbus-codegen
        --generate-c-code ${DBUS_GEN_PREFIX}
        --c-namespace OdlTb5
        --interface-prefix com.odinlink.
        ${DBUS_XML}
    DEPENDS ${DBUS_XML}
    COMMENT "Generating D-Bus bindings for daemon"
)

# Collect daemon source files
set(DAEMON_SOURCES
    src/odl_tb5_daemon_main.c
    src/odl_tb5_daemon_dbus.c
    src/odl_tb5_daemon_monitor.c
    src/odl_tb5_daemon_test.c
    src/odl_tb5_daemon_config.c
    src/odl_tb5_daemon_rccl_stats.c
    src/odl_tb5_daemon_sync.c
    src/odl_tb5_daemon_sync_proto.c
    src/odl_tb5_daemon_catalog.c
    src/odl_tb5_daemon_sysinfo.c
    # CLI source files (reused by daemon test executor, minus cli main)
    ${CMAKE_SOURCE_DIR}/cli/src/odl_tb5_cli_proto.c
    ${CMAKE_SOURCE_DIR}/cli/src/odl_tb5_cli_stats.c
    ${CMAKE_SOURCE_DIR}/cli/src/odl_tb5_cli_bandwidth.c
    ${CMAKE_SOURCE_DIR}/cli/src/odl_tb5_cli_latency.c
    ${CMAKE_SOURCE_DIR}/cli/src/odl_tb5_cli_latency_load.c
    ${CMAKE_SOURCE_DIR}/cli/src/odl_tb5_cli_mimo.c
    ${CMAKE_SOURCE_DIR}/cli/src/odl_tb5_cli_jitter.c
    ${CMAKE_SOURCE_DIR}/cli/src/odl_tb5_cli_server.c
    ${CMAKE_SOURCE_DIR}/cli/src/odl_tb5_cli_client.c
)

# Include directories
set(DAEMON_INCLUDES
    src
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_SOURCE_DIR}/lib/include
    ${CMAKE_SOURCE_DIR}/driver/uapi
    ${CMAKE_SOURCE_DIR}/cli/src
    ${GLIB2_INCLUDE_DIRS}
    ${GIO2_INCLUDE_DIRS}
    ${GIOUNIX_INCLUDE_DIRS}
)

# Link libraries
set(DAEMON_LIBS
    odl_tb5
    ${GLIB2_LIBRARIES}
    ${GIO2_LIBRARIES}
    ${GIOUNIX_LIBRARIES}
    pthread
    m
)

# Add FUSE3 support if available
if(FUSE3_FOUND)
    list(APPEND DAEMON_SOURCES src/odl_tb5_daemon_fuse.c)
    list(APPEND DAEMON_INCLUDES ${FUSE3_INCLUDE_DIRS})
    list(APPEND DAEMON_LIBS ${FUSE3_LIBRARIES})
endif()

# Add OpenSSL if available
if(OPENSSL_FOUND)
    list(APPEND DAEMON_INCLUDES ${OPENSSL_INCLUDE_DIRS})
    list(APPEND DAEMON_LIBS ${OPENSSL_LIBRARIES})
endif()

add_executable(odl_tb5_daemon ${DAEMON_SOURCES})

if(FUSE3_FOUND)
    target_compile_definitions(odl_tb5_daemon PRIVATE ODL_HAVE_FUSE3=1)
endif()
if(OPENSSL_FOUND)
    target_compile_definitions(odl_tb5_daemon PRIVATE ODL_HAVE_OPENSSL=1)
endif()

target_include_directories(odl_tb5_daemon PRIVATE ${DAEMON_INCLUDES})
target_link_libraries(odl_tb5_daemon ${DAEMON_LIBS})

include(GNUInstallDirs)
install(TARGETS odl_tb5_daemon
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT daemon
)
install(FILES data/odl-tb5-daemon.service
    DESTINATION lib/systemd/user
    COMPONENT daemon
)
install(FILES data/com.odinlink.Tb5Daemon.service
    DESTINATION ${CMAKE_INSTALL_DATADIR}/dbus-1/services
    COMPONENT daemon
)
