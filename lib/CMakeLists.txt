add_library(odl_tb5 SHARED
    src/odl_tb5_dev.c
    src/odl_tb5_xfer.c
    src/odl_tb5_peer.c
    src/odl_tb5_completion.c
    src/odl_tb5_stream.c
)

set_target_properties(odl_tb5 PROPERTIES
    VERSION   0.1.0
    SOVERSION 0
)

target_include_directories(odl_tb5 PUBLIC include)
target_include_directories(odl_tb5 PRIVATE ${CMAKE_SOURCE_DIR}/driver/uapi)

include(GNUInstallDirs)
install(TARGETS odl_tb5
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT library
)
install(DIRECTORY include/odl_tb5
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT library-dev
    FILES_MATCHING PATTERN "*.h"
)
