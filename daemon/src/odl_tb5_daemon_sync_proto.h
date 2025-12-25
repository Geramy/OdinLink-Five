/*
 * OdinLink TB5 Daemon - File Transfer Wire Protocol
 *
 * Copyright (c) 2025-2026 OdinLink Project
 */
#ifndef ODL_TB5_DAEMON_SYNC_PROTO_H
#define ODL_TB5_DAEMON_SYNC_PROTO_H

#include <odl_tb5/odl_tb5.h>
#include <stdint.h>
#include <stdbool.h>

#define ODL_SYNC_MAGIC       0x4F444C53
#define ODL_SYNC_VERSION     1
#define ODL_SYNC_CHUNK_SIZE  (512 * 1024)
#define ODL_SYNC_PATH_MAX    256

enum odl_sync_msg_type {
	ODL_SYNC_MSG_FILE_META     = 0x01,
	ODL_SYNC_MSG_FILE_DATA     = 0x02,
	ODL_SYNC_MSG_FILE_ACK      = 0x03,
	ODL_SYNC_MSG_FILE_DELETE   = 0x04,
	ODL_SYNC_MSG_DIR_CREATE    = 0x05,
	ODL_SYNC_MSG_DIR_DELETE    = 0x06,
	ODL_SYNC_MSG_SYNC_REQ      = 0x10,
	ODL_SYNC_MSG_LISTING_ENTRY = 0x11,
	ODL_SYNC_MSG_LISTING_END   = 0x12,
	ODL_SYNC_MSG_FETCH_REQ     = 0x20,
	ODL_SYNC_MSG_FETCH_RESP    = 0x21,
	ODL_SYNC_MSG_REMOVE_REQ    = 0x22,
	ODL_SYNC_MSG_REMOVE_ACK    = 0x23,
	ODL_SYNC_MSG_FILE_CHANGED  = 0x30,
	ODL_SYNC_MSG_FILE_REMOVED  = 0x31,
};

#define ODL_SYNC_ACK_OK        0
#define ODL_SYNC_ACK_CONFLICT  1
#define ODL_SYNC_ACK_ERROR     2
#define ODL_SYNC_ACK_REJECTED  3

struct odl_sync_header {
	uint32_t magic;
	uint32_t version;
	uint32_t type;
	uint32_t payload_len;
	uint32_t sequence;
	uint32_t reserved;
};

struct odl_sync_file_meta {
	struct odl_sync_header hdr;
	char     rel_path[ODL_SYNC_PATH_MAX];
	uint64_t file_size;
	uint64_t mtime_ns;
	uint32_t mode;
	uint32_t num_chunks;
	uint8_t  sha256[32];
};

struct odl_sync_file_data {
	struct odl_sync_header hdr;
	uint32_t chunk_index;
	uint32_t chunk_len;
};

struct odl_sync_file_ack {
	struct odl_sync_header hdr;
	char     rel_path[ODL_SYNC_PATH_MAX];
	uint32_t status;
	uint32_t reserved;
};

struct odl_sync_file_delete {
	struct odl_sync_header hdr;
	char     rel_path[ODL_SYNC_PATH_MAX];
	uint64_t mtime_ns;
};

struct odl_sync_dir_op {
	struct odl_sync_header hdr;
	char     rel_path[ODL_SYNC_PATH_MAX];
	uint64_t mtime_ns;
	uint32_t mode;
	uint32_t reserved;
};

struct odl_sync_listing_entry {
	struct odl_sync_header hdr;
	char     rel_path[ODL_SYNC_PATH_MAX];
	uint64_t file_size;
	uint64_t mtime_ns;
	uint32_t mode;
	uint32_t is_dir;
	uint8_t  sha256[32];
};

struct odl_sync_fetch_req {
	struct odl_sync_header hdr;
	char     rel_path[ODL_SYNC_PATH_MAX];
};

struct odl_sync_fetch_resp {
	struct odl_sync_header hdr;
	char     rel_path[ODL_SYNC_PATH_MAX];
	uint64_t file_size;
	uint64_t mtime_ns;
	uint32_t mode;
	uint32_t num_chunks;
	uint32_t status;
	uint32_t reserved;
	uint8_t  sha256[32];
};

#define ODL_FETCH_OK         0
#define ODL_FETCH_NOT_FOUND  1
#define ODL_FETCH_ERROR      2

struct odl_sync_remove_req {
	struct odl_sync_header hdr;
	char     rel_path[ODL_SYNC_PATH_MAX];
};

struct odl_sync_remove_ack {
	struct odl_sync_header hdr;
	char     rel_path[ODL_SYNC_PATH_MAX];
	uint32_t status;
	uint32_t reserved;
};

struct odl_sync_file_changed {
	struct odl_sync_header hdr;
	char     rel_path[ODL_SYNC_PATH_MAX];
	uint64_t file_size;
	uint64_t mtime_ns;
	uint32_t mode;
	uint32_t is_dir;
	uint8_t  sha256[32];
};

struct odl_sync_file_removed {
	struct odl_sync_header hdr;
	char     rel_path[ODL_SYNC_PATH_MAX];
	uint32_t is_dir;
	uint32_t reserved;
};

enum odl_file_location {
	ODL_FILE_LOCAL   = 0,
	ODL_FILE_REMOTE  = 1,
	ODL_FILE_CACHED  = 2,
	ODL_FILE_BOTH    = 3,
};

int odl_sync_send_file_meta(odl_tb5_t h, uint8_t sid, uint8_t dst,
			    uint32_t *seq,
			    const char *rel_path, uint64_t file_size,
			    uint64_t mtime_ns, uint32_t mode,
			    uint32_t num_chunks, const uint8_t sha256[32]);

int odl_sync_send_file_data(odl_tb5_t h, uint8_t sid, uint8_t dst,
			    uint32_t *seq,
			    uint32_t chunk_index, const void *data,
			    uint32_t chunk_len);

int odl_sync_send_file_ack(odl_tb5_t h, uint8_t sid, uint8_t dst,
			   uint32_t *seq,
			   const char *rel_path, uint32_t status);

int odl_sync_send_file_delete(odl_tb5_t h, uint8_t sid, uint8_t dst,
			      uint32_t *seq,
			      const char *rel_path, uint64_t mtime_ns);

int odl_sync_send_dir_create(odl_tb5_t h, uint8_t sid, uint8_t dst,
			     uint32_t *seq,
			     const char *rel_path, uint64_t mtime_ns,
			     uint32_t mode);

int odl_sync_send_dir_delete(odl_tb5_t h, uint8_t sid, uint8_t dst,
			     uint32_t *seq,
			     const char *rel_path, uint64_t mtime_ns);

int odl_sync_send_sync_req(odl_tb5_t h, uint8_t sid, uint8_t dst,
			   uint32_t *seq);

int odl_sync_send_listing_entry(odl_tb5_t h, uint8_t sid, uint8_t dst,
				uint32_t *seq,
				const char *rel_path, uint64_t file_size,
				uint64_t mtime_ns, uint32_t mode,
				bool is_dir, const uint8_t sha256[32]);

int odl_sync_send_listing_end(odl_tb5_t h, uint8_t sid, uint8_t dst,
			      uint32_t *seq);

int odl_sync_send_fetch_req(odl_tb5_t h, uint8_t sid, uint8_t dst,
			    uint32_t *seq,
			    const char *rel_path);

int odl_sync_send_fetch_resp(odl_tb5_t h, uint8_t sid, uint8_t dst,
			     uint32_t *seq,
			     const char *rel_path, uint64_t file_size,
			     uint64_t mtime_ns, uint32_t mode,
			     uint32_t num_chunks, uint32_t status,
			     const uint8_t sha256[32]);

int odl_sync_send_remove_req(odl_tb5_t h, uint8_t sid, uint8_t dst,
			     uint32_t *seq,
			     const char *rel_path);

int odl_sync_send_remove_ack(odl_tb5_t h, uint8_t sid, uint8_t dst,
			     uint32_t *seq,
			     const char *rel_path, uint32_t status);

int odl_sync_send_file_changed(odl_tb5_t h, uint8_t sid, uint8_t dst,
			       uint32_t *seq,
			       const char *rel_path, uint64_t file_size,
			       uint64_t mtime_ns, uint32_t mode,
			       bool is_dir, const uint8_t sha256[32]);

int odl_sync_send_file_removed(odl_tb5_t h, uint8_t sid, uint8_t dst,
			       uint32_t *seq,
			       const char *rel_path, bool is_dir);

int odl_sync_recv_msg(odl_tb5_t h, uint8_t sid, void *buf, size_t buf_size,
		      uint32_t *out_type);

int odl_sync_sha256_file(const char *path, uint8_t out[32]);

#endif /* ODL_TB5_DAEMON_SYNC_PROTO_H */
