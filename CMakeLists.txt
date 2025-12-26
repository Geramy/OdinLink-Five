cmake_minimum_required(VERSION 3.10)
project(tb5-rccl-project)

# Include third_party headers
include_directories(third_party)

# Build kernel module
add_custom_target(kernel_module ALL
    COMMAND make all
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/kernel
    COMMENT "Building Thunderbolt 5 kernel module"
)

# Add subdirectories
add_subdirectory(userspace)
add_subdirectory(plugin)

# Build test program
add_executable(tb5_test test_main.c)
target_include_directories(tb5_test PRIVATE ${CMAKE_SOURCE_DIR}/third_party)
target_link_libraries(tb5_test tb5_ring rccl_net_tb5 dl)
