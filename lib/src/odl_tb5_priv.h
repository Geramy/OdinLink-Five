/*
 * OdinLink — Internal Library Plumbing
 *
 * The handle struct that every library function uses behind the scenes.
 * Holds the device fd, mmap'd buffer pointers, and buffer sizes.
 * NOT exposed to users — this is implementation detail.
 */
#ifndef ODL_TB5_PRIV_H
#define ODL_TB5_PRIV_H

#include <odl_tb5/odl_tb5.h>
#include <stddef.h>

struct odl_tb5_handle {
	int    fd;
	void  *tx_bufs[2];
	void  *rx_bufs[2];
	size_t tx_buf_size;
	size_t rx_buf_size;
	int    tx_back;
	int    rx_back;
};

#endif /* ODL_TB5_PRIV_H */
