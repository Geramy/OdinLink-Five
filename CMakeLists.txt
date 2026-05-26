cmake_minimum_required(VERSION 3.10)
project(odinlink-five VERSION 0.1.0 LANGUAGES C)

include(GNUInstallDirs)

# Include third_party headers
include_directories(third_party/rccl)

# =============================================================================
# Consolidated dependency check
# =============================================================================
set(ODL_MISSING_DEPS "")

find_program(KERNEL_GCC NAMES gcc-15 gcc-14 gcc QUIET)
if(NOT KERNEL_GCC)
    list(APPEND ODL_MISSING_DEPS "gcc-14")
endif()

find_program(PKG_CONFIG_EXE NAMES pkg-config QUIET)
if(NOT PKG_CONFIG_EXE)
    list(APPEND ODL_MISSING_DEPS "pkg-config")
endif()

# gdbus-codegen is required only when building the daemon (checked in daemon/CMakeLists.txt)
# We still probe it here for the missing deps warning, but don't fail on it unless
# BUILD_DAEMON is ON.
option(BUILD_DAEMON "Build OdinLink TB5 system daemon" ON)
option(BUILD_TRAY   "Build OdinLink TB5 tray application" ON)
find_program(GDBUS_CODEGEN_EXE NAMES gdbus-codegen QUIET)
if(NOT GDBUS_CODEGEN_EXE AND BUILD_DAEMON)
    list(APPEND ODL_MISSING_DEPS "libglib2.0-dev-bin")
endif()

# Kernel module build — optional. Skip header check when building userspace only.
option(BUILD_KERNEL_MODULE "Build the odl_tb5 kernel module" ON)
if(BUILD_KERNEL_MODULE AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    execute_process(COMMAND uname -r OUTPUT_VARIABLE KERNEL_RELEASE OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT EXISTS "/lib/modules/${KERNEL_RELEASE}/build")
        list(APPEND ODL_MISSING_DEPS "linux-headers-${KERNEL_RELEASE}")
    endif()
endif()

if(PKG_CONFIG_EXE)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(_CHK_GLIB2 QUIET glib-2.0>=2.56)
        if(NOT _CHK_GLIB2_FOUND)
            list(APPEND ODL_MISSING_DEPS "libglib2.0-dev")
        endif()
        pkg_check_modules(_CHK_GTK3 QUIET gtk+-3.0>=3.22)
        if(NOT _CHK_GTK3_FOUND)
            list(APPEND ODL_MISSING_DEPS "libgtk-3-dev")
        endif()
        pkg_check_modules(_CHK_APPIND QUIET ayatana-appindicator3-0.1)
        if(NOT _CHK_APPIND_FOUND)
            list(APPEND ODL_MISSING_DEPS "libayatana-appindicator3-dev")
        endif()
        # Optional
        pkg_check_modules(_CHK_FUSE3 QUIET fuse3>=3.1)
        if(NOT _CHK_FUSE3_FOUND)
            message(STATUS "Optional: libfuse3-dev not found (distributed file access disabled)")
        endif()
        pkg_check_modules(_CHK_OPENSSL QUIET libcrypto)
        if(NOT _CHK_OPENSSL_FOUND)
            message(STATUS "Optional: libssl-dev not found (SHA-256 sync disabled)")
        endif()
    endif()
else()
    list(APPEND ODL_MISSING_DEPS "libglib2.0-dev" "libgtk-3-dev" "libayatana-appindicator3-dev")
endif()

list(REMOVE_DUPLICATES ODL_MISSING_DEPS)
if(ODL_MISSING_DEPS)
    string(REPLACE ";" " " _DEPS_STR "${ODL_MISSING_DEPS}")
    message(WARNING
        "\n"
        "============================================================\n"
        "  Missing build dependencies!\n"
        "\n"
        "  Run:\n"
        "    sudo apt install ${_DEPS_STR}\n"
        "\n"
        "  Then re-run cmake.\n"
        "============================================================\n")
    # Fail only if a required dep is missing (not optional fuse3/openssl)
    if(BUILD_KERNEL_MODULE)
        if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
           (NOT KERNEL_GCC OR NOT PKG_CONFIG_EXE OR NOT GDBUS_CODEGEN_EXE
            OR NOT EXISTS "/lib/modules/${KERNEL_RELEASE}/build"))
            message(FATAL_ERROR "Cannot continue without required dependencies.")
        endif()
    else()
        if(NOT PKG_CONFIG_EXE)
            message(FATAL_ERROR "Cannot continue without required dependencies.")
        endif()
    endif()
endif()
if(NOT ODL_MISSING_DEPS)
    message(STATUS "All build dependencies satisfied")
endif()

# =============================================================================
# Kernel module
# =============================================================================

# Build kernel module (odl_tb5.ko) — not part of ALL so CPack preinstall
# doesn't try to rebuild it. Build explicitly with: make driver
add_custom_target(driver
    COMMAND make all CC=${KERNEL_GCC}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/driver
    COMMENT "Building OdinLink TB5 kernel module (odl_tb5.ko)"
)

# Userspace library (libodl_tb5.so)
add_subdirectory(lib)

# RCCL plugin (librccl_net_odl_tb5.so)
add_subdirectory(rccl)

# NCCL plugin (libnccl-net-ODL_TB5.so) — for CUDA/NVIDIA/PyTorch
option(BUILD_NCCL "Build OdinLink TB5 NCCL network plugin" ON)
if(BUILD_NCCL)
    add_subdirectory(nccl)
endif()

# CLI tool (odl_tb5_cli)
add_subdirectory(cli)

# Test suite (odl_tb5_test)
add_subdirectory(tests)

# Optional: System daemon and tray application (options defined above)

# Auto-disable if dependencies not found
if(BUILD_DAEMON AND NOT _CHK_GLIB2_FOUND)
    message(WARNING "glib-2.0 / gio-2.0 not found -- disabling daemon build.")
    set(BUILD_DAEMON OFF)
endif()

if(BUILD_TRAY AND (NOT _CHK_GTK3_FOUND OR NOT _CHK_APPIND_FOUND))
    message(WARNING "GTK3 or ayatana-appindicator3 not found -- disabling tray build.")
    set(BUILD_TRAY OFF)
endif()

if(BUILD_DAEMON)
    add_subdirectory(daemon)
endif()
if(BUILD_TRAY)
    add_subdirectory(tray)
endif()

# Verbs provider plugin (libibverbs-compatible RDMA)
option(BUILD_VERBS "Build OdinLink TB5 verbs provider plugin" ON)
if(BUILD_VERBS)
    add_subdirectory(verbs)
endif()

message(STATUS "BUILD_DAEMON: ${BUILD_DAEMON}")
message(STATUS "BUILD_TRAY:   ${BUILD_TRAY}")
message(STATUS "BUILD_NCCL:   ${BUILD_NCCL}")
message(STATUS "BUILD_VERBS:  ${BUILD_VERBS}")

# =============================================================================
# DKMS kernel module install (source files for DKMS to compile per-kernel)
# =============================================================================
set(DKMS_SRC_DIR "src/odl-tb5-${PROJECT_VERSION}")

install(FILES
    driver/odl_tb5_service.c
    driver/odl_tb5_chardev.c
    driver/odl_tb5_ring_dma.c
    driver/odl_tb5_proto.c
    driver/odl_tb5_loopback.c
    driver/odl_tb5_core.h
    driver/Kbuild
    driver/Makefile
    DESTINATION ${DKMS_SRC_DIR}
    COMPONENT dkms
)
install(FILES driver/uapi/odl_tb5_uapi.h
    DESTINATION ${DKMS_SRC_DIR}/uapi
    COMPONENT dkms
)

# Configure dkms.conf with project version
configure_file(
    packaging/dkms.conf
    ${CMAKE_CURRENT_BINARY_DIR}/dkms.conf
    @ONLY
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/dkms.conf
    DESTINATION ${DKMS_SRC_DIR}
    COMPONENT dkms
)
install(FILES driver/71-odl-tb5.rules
    DESTINATION /etc/udev/rules.d
    COMPONENT dkms
)

# =============================================================================
# CPack - component-based .deb generation
# =============================================================================
set(CPACK_GENERATOR "DEB")
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_DEBIAN_ENABLE_COMPONENT_DEPENDS ON)
set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")

# Package metadata
set(CPACK_PACKAGE_VENDOR "OdinLink Project")
set(CPACK_PACKAGE_CONTACT "OdinLink Project <odinlink@example.com>")
set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/OdinLink/OdinLink-Five")

# --- dkms ---
set(CPACK_DEBIAN_DKMS_PACKAGE_NAME "odl-tb5-dkms")
set(CPACK_DEBIAN_DKMS_DESCRIPTION
    "OdinLink Thunderbolt 5 kernel driver (DKMS)")
set(CPACK_DEBIAN_DKMS_PACKAGE_DEPENDS
    "dkms, gcc-14 | gcc-15 | gcc")
set(CPACK_DEBIAN_DKMS_PACKAGE_ARCHITECTURE "all")

# Configure packaging scripts (substitute @PROJECT_VERSION@)
configure_file(packaging/dkms-postinst.sh ${CMAKE_BINARY_DIR}/packaging/dkms-postinst.sh @ONLY)
configure_file(packaging/dkms-prerm.sh    ${CMAKE_BINARY_DIR}/packaging/dkms-prerm.sh    @ONLY)
configure_file(packaging/daemon-postinst.sh ${CMAKE_BINARY_DIR}/packaging/daemon-postinst.sh @ONLY)

set(CPACK_DEBIAN_DKMS_PACKAGE_CONTROL_EXTRA
    "${CMAKE_BINARY_DIR}/packaging/dkms-postinst.sh;${CMAKE_BINARY_DIR}/packaging/dkms-prerm.sh")

# --- library ---
set(CPACK_DEBIAN_LIBRARY_PACKAGE_NAME "libodl-tb5-0")
set(CPACK_DEBIAN_LIBRARY_DESCRIPTION
    "OdinLink Thunderbolt 5 userspace library")
set(CPACK_DEBIAN_LIBRARY_PACKAGE_DEPENDS "libc6")

# --- library-dev ---
set(CPACK_DEBIAN_LIBRARY-DEV_PACKAGE_NAME "libodl-tb5-dev")
set(CPACK_DEBIAN_LIBRARY-DEV_DESCRIPTION
    "OdinLink Thunderbolt 5 development headers")
set(CPACK_DEBIAN_LIBRARY-DEV_PACKAGE_DEPENDS "libodl-tb5-0 (= ${PROJECT_VERSION})")

# --- cli ---
set(CPACK_DEBIAN_CLI_PACKAGE_NAME "odl-tb5-cli")
set(CPACK_DEBIAN_CLI_DESCRIPTION
    "OdinLink Thunderbolt 5 CLI test tool")
set(CPACK_DEBIAN_CLI_PACKAGE_DEPENDS "libodl-tb5-0 (>= ${PROJECT_VERSION})")

# --- daemon ---
set(CPACK_DEBIAN_DAEMON_PACKAGE_NAME "odl-tb5-daemon")
set(CPACK_DEBIAN_DAEMON_DESCRIPTION
    "OdinLink Thunderbolt 5 background daemon")
set(CPACK_DEBIAN_DAEMON_PACKAGE_DEPENDS
    "libodl-tb5-0 (>= ${PROJECT_VERSION}), libglib2.0-0 (>= 2.56)")
set(CPACK_DEBIAN_DAEMON_PACKAGE_CONTROL_EXTRA
    "${CMAKE_BINARY_DIR}/packaging/daemon-postinst.sh")

# --- tray ---
set(CPACK_DEBIAN_TRAY_PACKAGE_NAME "odl-tb5-tray")
set(CPACK_DEBIAN_TRAY_DESCRIPTION
    "OdinLink Thunderbolt 5 system tray application")
set(CPACK_DEBIAN_TRAY_PACKAGE_DEPENDS
    "odl-tb5-daemon (>= ${PROJECT_VERSION}), libgtk-3-0 (>= 3.22), libayatana-appindicator3-1")

# --- rccl ---
set(CPACK_DEBIAN_RCCL_PACKAGE_NAME "odl-tb5-rccl")
set(CPACK_DEBIAN_RCCL_DESCRIPTION
    "OdinLink Thunderbolt 5 RCCL network plugin (AMD ROCm)")
set(CPACK_DEBIAN_RCCL_PACKAGE_DEPENDS "libodl-tb5-0 (>= ${PROJECT_VERSION})")

# --- nccl ---
set(CPACK_DEBIAN_NCCL_PACKAGE_NAME "odl-tb5-nccl")
set(CPACK_DEBIAN_NCCL_DESCRIPTION
    "OdinLink Thunderbolt 5 NCCL network plugin (NVIDIA CUDA)")
set(CPACK_DEBIAN_NCCL_PACKAGE_DEPENDS "libodl-tb5-0 (>= ${PROJECT_VERSION})")

include(CPack)

# Define components
cpack_add_component(dkms        DISPLAY_NAME "Kernel Driver (DKMS)")
cpack_add_component(library     DISPLAY_NAME "Userspace Library")
cpack_add_component(library-dev DISPLAY_NAME "Development Headers"  DEPENDS library)
cpack_add_component(cli         DISPLAY_NAME "CLI Test Tool"        DEPENDS library)
cpack_add_component(daemon      DISPLAY_NAME "Background Daemon"    DEPENDS library)
cpack_add_component(tray        DISPLAY_NAME "System Tray App"      DEPENDS daemon)
cpack_add_component(rccl        DISPLAY_NAME "RCCL Plugin (AMD)"    DEPENDS library)
cpack_add_component(nccl        DISPLAY_NAME "NCCL Plugin (NVIDIA)"  DEPENDS library)

# =============================================================================
# Package target: creates 3 self-contained .deb installers
#
#   cmake --build . --target meta-packages
#
# Produces:
#   odl-tb5-minimal_0.1.0_amd64.deb (GPU cluster: dkms+lib+rccl+nccl)
#   odl-tb5-server_0.1.0_amd64.deb  (headless:    dkms+lib+cli+daemon+rccl+nccl)
#   odl-tb5-full_0.1.0_amd64.deb    (full:        everything)
# =============================================================================

# Configure the build script with version
configure_file(
    ${CMAKE_SOURCE_DIR}/packaging/build-meta-debs.sh.in
    ${CMAKE_BINARY_DIR}/build-meta-debs.sh
    @ONLY
)

add_custom_target(meta-packages
    COMMAND bash ${CMAKE_BINARY_DIR}/build-meta-debs.sh ${CMAKE_BINARY_DIR}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Building OdinLink TB5 packages (minimal, server, full)"
)
