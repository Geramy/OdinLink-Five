find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK3   REQUIRED gtk+-3.0>=3.22)
pkg_check_modules(APPIND REQUIRED ayatana-appindicator3-0.1)
pkg_check_modules(GIO2   REQUIRED gio-2.0>=2.56)

# Generate D-Bus C bindings (client proxy)
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
    COMMENT "Generating D-Bus bindings for tray"
)

add_executable(odl_tb5_tray
    src/odl_tb5_tray_main.c
    src/odl_tb5_tray_dbus.c
    src/odl_tb5_tray_menu.c
    src/odl_tb5_tray_peers.c
    src/odl_tb5_tray_tests.c
    src/odl_tb5_tray_test_parse.c
    src/odl_tb5_tray_test_charts.c
    src/odl_tb5_tray_sync.c
    src/odl_tb5_tray_rccl.c
    src/odl_tb5_tray_files.c
)

target_include_directories(odl_tb5_tray PRIVATE
    src
    ${CMAKE_CURRENT_BINARY_DIR}
    ${GTK3_INCLUDE_DIRS}
    ${APPIND_INCLUDE_DIRS}
    ${GIO2_INCLUDE_DIRS}
)

target_link_libraries(odl_tb5_tray
    ${GTK3_LIBRARIES}
    ${APPIND_LIBRARIES}
    ${GIO2_LIBRARIES}
)

# Pass icon directory so AppIndicator finds our custom SVGs during development
target_compile_definitions(odl_tb5_tray PRIVATE
    ODL_TRAY_ICON_DIR="${CMAKE_CURRENT_SOURCE_DIR}/icons"
)

include(GNUInstallDirs)
install(TARGETS odl_tb5_tray
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT tray
)
install(FILES data/odl-tb5-tray.desktop
    DESTINATION ${CMAKE_INSTALL_DATADIR}/applications
    COMPONENT tray
)
install(FILES data/odl-tb5-tray.desktop
    DESTINATION /etc/xdg/autostart
    COMPONENT tray
)
install(FILES
    icons/odl_tb5_connected.svg
    icons/odl_tb5_disconnected.svg
    icons/odl_tb5_syncing.svg
    icons/odl_tb5_error.svg
    DESTINATION ${CMAKE_INSTALL_DATADIR}/icons/hicolor/scalable/apps
    COMPONENT tray
)
