add_custom_target(tb5_dma_ring.ko ALL
    COMMAND make -C /lib/modules/`uname -r`/build M=${CMAKE_CURRENT_SOURCE_DIR} obj-m=tb5_dma_ring.o modules
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building Thunderbolt 5 DMA ring kernel module"
)

add_custom_target(insmod
    COMMAND sudo insmod tb5_dma_ring.ko
    DEPENDS tb5_dma_ring.ko
    COMMENT "Loading kernel module"
)

add_custom_target(rmmod
    COMMAND sudo rmmod tb5_dma_ring
    COMMENT "Unloading kernel module"
)
