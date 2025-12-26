include_directories(include)
add_library(tb5_ring SHARED src/tb5_ring.c)
target_include_directories(tb5_ring PUBLIC include)
