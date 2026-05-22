# OdinLink Verbs Provider — libibverbs-compatible RDMA provider plugin
#
# Builds as a standalone verbs library that provides:
#   - Standard ibv_* API for OdinLink-Five devices
#   - Zero-copy GPU via ibv_reg_dmabuf_mr → dmabuf fd
#   - Async completion via workqueue worker + eventfd
#
# Build modes:
#   1. SHARED: libodl_tb5_verbs.so → apps link with -lodl_tb5_verbs
#   2. Provider plugin: libodl_tb5-rdmav34.so → /usr/lib/.../libibverbs/
#
# Dependencies: libodl_tb5, libibverbs, pthreads

cmake_minimum_required(VERSION 3.10)

option(ODL_VERBS_BUILD_PROVIDER_PLUGIN "Build libibverbs provider plugin" ON)

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(IBVERBS QUIET libibverbs)
endif()

if(NOT IBVERBS_FOUND)
    message(STATUS "libibverbs not found -- verbs provider disabled. "
                   "Install libibverbs-dev (rdma-core) to enable.")
    return()
endif()

# Determine provider plugin install directory
if(ODL_VERBS_BUILD_PROVIDER_PLUGIN)
    foreach(DIR
        "/usr/lib/aarch64-linux-gnu/libibverbs"
        "/usr/lib/x86_64-linux-gnu/libibverbs"
        "/usr/lib64/libibverbs"
        "/usr/lib/libibverbs"
        "${CMAKE_INSTALL_LIBDIR}/libibverbs")
        if(IS_DIRECTORY "${DIR}")
            set(IBVERBS_PROVIDER_DIR "${DIR}")
            break()
        endif()
    endforeach()
    if(NOT IBVERBS_PROVIDER_DIR)
        set(IBVERBS_PROVIDER_DIR "${CMAKE_INSTALL_LIBDIR}/libibverbs")
    endif()
endif()

# Sources
set(VERBS_SOURCES
    src/odl_tb5_verbs_main.c
    src/odl_tb5_verbs_device.c
    src/odl_tb5_verbs_pd.c
    src/odl_tb5_verbs_mr.c
    src/odl_tb5_verbs_cq.c
    src/odl_tb5_verbs_qp.c
    src/odl_tb5_verbs_ops.c
)

# Standalone shared library
add_library(odl_tb5_verbs SHARED ${VERBS_SOURCES})

target_include_directories(odl_tb5_verbs PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/lib/include
    ${IBVERBS_INCLUDE_DIRS}
)

target_link_libraries(odl_tb5_verbs
    odl_tb5
    ${IBVERBS_LIBRARIES}
    pthread
    dl
    m
)

set_target_properties(odl_tb5_verbs PROPERTIES
    VERSION    0.1.0
    SOVERSION  0
    INSTALL_RPATH "${CMAKE_INSTALL_LIBDIR}"
    BUILD_WITH_INSTALL_RPATH ON
)

# If provider plugin is enabled, also build the rdma-core plugin
if(ODL_VERBS_BUILD_PROVIDER_PLUGIN AND IBVERBS_PROVIDER_DIR)
    add_library(odl_tb5_verbs_provider SHARED ${VERBS_SOURCES})

    target_include_directories(odl_tb5_verbs_provider PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/lib/include
        ${IBVERBS_INCLUDE_DIRS}
    )

    target_link_libraries(odl_tb5_verbs_provider
        odl_tb5
        ${IBVERBS_LIBRARIES}
        pthread
        dl
        m
    )

    set_target_properties(odl_tb5_verbs_provider PROPERTIES
        LIBRARY_OUTPUT_NAME "odl_tb5-rdmav34"
        VERSION    0.1.0
        SOVERSION  34
        INSTALL_RPATH "${CMAKE_INSTALL_LIBDIR}"
        BUILD_WITH_INSTALL_RPATH ON
    )

    install(TARGETS odl_tb5_verbs_provider
        LIBRARY DESTINATION ${IBVERBS_PROVIDER_DIR}
        COMPONENT verbs
    )
endif()

install(TARGETS odl_tb5_verbs
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT verbs
)

install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/odl_tb5_verbs_wrapper.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/odl_tb5
    COMPONENT library-dev
)

message(STATUS "BUILD_VERBS: ON")
message(STATUS "  Provider plugin: ${ODL_VERBS_BUILD_PROVIDER_PLUGIN}")
message(STATUS "  Provider dir:    ${IBVERBS_PROVIDER_DIR}")
