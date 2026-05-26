/*
 * OdinLink — Daemon: Sync Wire Protocol (Messages Between Peers)
 *
 * Serializes and deserializes the messages the sync engine exchanges
 * with the peer: file change notifications, chunk requests, checksums,
 * and transfer completions. Runs over OdinLink stream 2.
 */
#include "odl_tb5_daemon_sync_proto.h"

#include <odl_tb5/odl_tb5.h>
#include <openssl/evp.h>
#include <openssl/hmac.h>

/* Pre-shared key for HMAC-SHA256 message authentication.
 * Override at build time with -DODL_SYNC_HMAC_KEY="..." */
#ifndef ODL_SYNC_HMAC_KEY
#define ODL_SYNC_HMAC_KEY  "odinlink-sync-default-key"
#endif

#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* Fill the common sync header fields and stamp a truncated HMAC-SHA256
 * authentication tag into the reserved field. */
static void fill_header(struct odl_sync_header *hdr, uint32_t type,
			uint32_t payload_len, uint32_t seq)
{
	uint8_t digest[32];
	unsigned int dlen = sizeof(digest);

	memset(hdr, 0, sizeof(*hdr));
	hdr->magic       = ODL_SYNC_MAGIC;
	hdr->version     = ODL_SYNC_VERSION;
	hdr->type        = type;
	hdr->payload_len = payload_len;
	hdr->sequence    = seq;
	/* Compute HMAC over the authenticated fields (all except reserved). */
	HMAC(EVP_sha256(), ODL_SYNC_HMAC_KEY, sizeof(ODL_SYNC_HMAC_KEY) - 1,
	     (const uint8_t *)hdr, sizeof(*hdr) - sizeof(hdr->reserved),
	     digest, &dlen);
	memcpy(&hdr->reserved, digest, sizeof(hdr->reserved));
}

/* Send an assembled message via stream. */
static int send_ctrl(odl_tb5_t h, uint8_t sid, uint8_t dst,
		     const void *msg, size_t msg_len)
{
	return odl_tb5_stream_send(h, sid, dst, msg, msg_len);
}

int odl_sync_send_file_meta(odl_tb5_t h, uint8_t sid, uint8_t dst,
			    uint32_t *seq,
			    const char *rel_path, uint64_t file_size,
			    uint64_t mtime_ns, uint32_t mode,
			    uint32_t num_chunks, const uint8_t sha256[32])
{
	struct odl_sync_file_meta msg;

	memset(&msg, 0, sizeof(msg));
	(*seq)++;
	fill_header(&msg.hdr, ODL_SYNC_MSG_FILE_META,
		    sizeof(msg) - sizeof(msg.hdr), *seq);

	snprintf(msg.rel_path, sizeof(msg.rel_path), "%s", rel_path);
	msg.file_size   = file_size;
	msg.mtime_ns    = mtime_ns;
	msg.mode        = mode;
	msg.num_chunks  = num_chunks;
	memcpy(msg.sha256, sha256, 32);

	return send_ctrl(h, sid, dst, &msg, sizeof(msg));
}

int odl_sync_send_file_data(odl_tb5_t h, uint8_t sid, uint8_t dst,
			    uint32_t *seq,
			    uint32_t chunk_index, const void *data,
			    uint32_t chunk_len)
{
	struct odl_sync_file_data hdr;
	size_t total_len;
	uint8_t *buf;
	int ret;

	memset(&hdr, 0, sizeof(hdr));
	(*seq)++;
	fill_header(&hdr.hdr, ODL_SYNC_MSG_FILE_DATA,
		    sizeof(hdr) - sizeof(hdr.hdr) + chunk_len, *seq);
	hdr.chunk_index = chunk_index;
	hdr.chunk_len   = chunk_len;

	total_len = sizeof(hdr) + chunk_len;

	buf = malloc(total_len);
	if (!buf)
		return -ENOMEM;

	memcpy(buf, &hdr, sizeof(hdr));
	memcpy(buf + sizeof(hdr), data, chunk_len);

	ret = odl_tb5_stream_send(h, sid, dst, buf, total_len);

	free(buf);
	return ret;
}

int odl_sync_send_file_ack(odl_tb5_t h, uint8_t sid, uint8_t dst,
			   uint32_t *seq,
			   const char *rel_path, uint32_t status)
{
	struct odl_sync_file_ack msg;

	memset(&msg, 0, sizeof(msg));
	(*seq)++;
	fill_header(&msg.hdr, ODL_SYNC_MSG_FILE_ACK,
		    sizeof(msg) - sizeof(msg.hdr), *seq);

	snprintf(msg.rel_path, sizeof(msg.rel_path), "%s", rel_path);
	msg.status = status;

	return send_ctrl(h, sid, dst, &msg, sizeof(msg));
}

int odl_sync_send_file_delete(odl_tb5_t h, uint8_t sid, uint8_t dst,
			      uint32_t *seq,
			      const char *rel_path, uint64_t mtime_ns)
{
	struct odl_sync_file_delete msg;

	memset(&msg, 0, sizeof(msg));
	(*seq)++;
	fill_header(&msg.hdr, ODL_SYNC_MSG_FILE_DELETE,
		    sizeof(msg) - sizeof(msg.hdr), *seq);

	snprintf(msg.rel_path, sizeof(msg.rel_path), "%s", rel_path);
	msg.mtime_ns = mtime_ns;

	return send_ctrl(h, sid, dst, &msg, sizeof(msg));
}

int odl_sync_send_dir_create(odl_tb5_t h, uint8_t sid, uint8_t dst,
			     uint32_t *seq,
			     const char *rel_path, uint64_t mtime_ns,
			     uint32_t mode)
{
	struct odl_sync_dir_op msg;

	memset(&msg, 0, sizeof(msg));
	(*seq)++;
	fill_header(&msg.hdr, ODL_SYNC_MSG_DIR_CREATE,
		    sizeof(msg) - sizeof(msg.hdr), *seq);

	snprintf(msg.rel_path, sizeof(msg.rel_path), "%s", rel_path);
	msg.mtime_ns = mtime_ns;
	msg.mode     = mode;

	return send_ctrl(h, sid, dst, &msg, sizeof(msg));
}

int odl_sync_send_dir_delete(odl_tb5_t h, uint8_t sid, uint8_t dst,
			     uint32_t *seq,
			     const char *rel_path, uint64_t mtime_ns)
{
	struct odl_sync_dir_op msg;

	memset(&msg, 0, sizeof(msg));
	(*seq)++;
	fill_header(&msg.hdr, ODL_SYNC_MSG_DIR_DELETE,
		    sizeof(msg) - sizeof(msg.hdr), *seq);

	snprintf(msg.rel_path, sizeof(msg.rel_path), "%s", rel_path);
	msg.mtime_ns = mtime_ns;

	return send_ctrl(h, sid, dst, &msg, sizeof(msg));
}

int odl_sync_send_sync_req(odl_tb5_t h, uint8_t sid, uint8_t dst,
			   uint32_t *seq)
{
	struct odl_sync_header msg;

	(*seq)++;
	fill_header(&msg, ODL_SYNC_MSG_SYNC_REQ, 0, *seq);

	return send_ctrl(h, sid, dst, &msg, sizeof(msg));
}

int odl_sync_send_listing_entry(odl_tb5_t h, uint8_t sid, uint8_t dst,
				uint32_t *seq,
				const char *rel_path, uint64_t file_size,
				uint64_t mtime_ns, uint32_t mode,
				bool is_dir, const uint8_t sha256[32])
{
	struct odl_sync_listing_entry msg;

	memset(&msg, 0, sizeof(msg));
	(*seq)++;
	fill_header(&msg.hdr, ODL_SYNC_MSG_LISTING_ENTRY,
		    sizeof(msg) - sizeof(msg.hdr), *seq);

	strncpy(msg.rel_path, rel_path, ODL_SYNC_PATH_MAX - 1);
	msg.rel_path[ODL_SYNC_PATH_MAX - 1] = '\0';
	msg.file_size = file_size;
	msg.mtime_ns  = mtime_ns;
	msg.mode      = mode;
	msg.is_dir    = is_dir ? 1 : 0;
	if (sha256)
		memcpy(msg.sha256, sha256, 32);

	return send_ctrl(h, sid, dst, &msg, sizeof(msg));
}

int odl_sync_send_listing_end(odl_tb5_t h, uint8_t sid, uint8_t dst,
			      uint32_t *seq)
{
	struct odl_sync_header msg;

	(*seq)++;
	fill_header(&msg, ODL_SYNC_MSG_LISTING_END, 0, *seq);

	return send_ctrl(h, sid, dst, &msg, sizeof(msg));
}

int odl_sync_send_fetch_req(odl_tb5_t h, uint8_t sid, uint8_t dst,
			    uint32_t *seq,
			    const char *rel_path)
{
	struct odl_sync_fetch_req msg;

	memset(&msg, 0, sizeof(msg));
	(*seq)++;
	fill_header(&msg.hdr, ODL_SYNC_MSG_FETCH_REQ,
		    sizeof(msg) - sizeof(msg.hdr), *seq);

	strncpy(msg.rel_path, rel_path, ODL_SYNC_PATH_MAX - 1);
	msg.rel_path[ODL_SYNC_PATH_MAX - 1] = '\0';

	return send_ctrl(h, sid, dst, &msg, sizeof(msg));
}

int odl_sync_send_fetch_resp(odl_tb5_t h, uint8_t sid, uint8_t dst,
			     uint32_t *seq,
			     const char *rel_path, uint64_t file_size,
			     uint64_t mtime_ns, uint32_t mode,
			     uint32_t num_chunks, uint32_t status,
			     const uint8_t sha256[32])
{
	struct odl_sync_fetch_resp msg;

	memset(&msg, 0, sizeof(msg));
	(*seq)++;
	fill_header(&msg.hdr, ODL_SYNC_MSG_FETCH_RESP,
		    sizeof(msg) - sizeof(msg.hdr), *seq);

	strncpy(msg.rel_path, rel_path, ODL_SYNC_PATH_MAX - 1);
	msg.rel_path[ODL_SYNC_PATH_MAX - 1] = '\0';
	msg.file_size   = file_size;
	msg.mtime_ns    = mtime_ns;
	msg.mode        = mode;
	msg.num_chunks  = num_chunks;
	msg.status      = status;
	if (sha256)
		memcpy(msg.sha256, sha256, 32);

	return send_ctrl(h, sid, dst, &msg, sizeof(msg));
}

int odl_sync_send_remove_req(odl_tb5_t h, uint8_t sid, uint8_t dst,
			     uint32_t *seq,
			     const char *rel_path)
{
	struct odl_sync_remove_req msg;

	memset(&msg, 0, sizeof(msg));
	(*seq)++;
	fill_header(&msg.hdr, ODL_SYNC_MSG_REMOVE_REQ,
		    sizeof(msg) - sizeof(msg.hdr), *seq);

	strncpy(msg.rel_path, rel_path, ODL_SYNC_PATH_MAX - 1);
	msg.rel_path[ODL_SYNC_PATH_MAX - 1] = '\0';

	return send_ctrl(h, sid, dst, &msg, sizeof(msg));
}

int odl_sync_send_remove_ack(odl_tb5_t h, uint8_t sid, uint8_t dst,
			     uint32_t *seq,
			     const char *rel_path, uint32_t status)
{
	struct odl_sync_remove_ack msg;

	memset(&msg, 0, sizeof(msg));
	(*seq)++;
	fill_header(&msg.hdr, ODL_SYNC_MSG_REMOVE_ACK,
		    sizeof(msg) - sizeof(msg.hdr), *seq);

	strncpy(msg.rel_path, rel_path, ODL_SYNC_PATH_MAX - 1);
	msg.rel_path[ODL_SYNC_PATH_MAX - 1] = '\0';
	msg.status = status;

	return send_ctrl(h, sid, dst, &msg, sizeof(msg));
}

int odl_sync_send_file_changed(odl_tb5_t h, uint8_t sid, uint8_t dst,
			       uint32_t *seq,
			       const char *rel_path, uint64_t file_size,
			       uint64_t mtime_ns, uint32_t mode,
			       bool is_dir, const uint8_t sha256[32])
{
	struct odl_sync_file_changed msg;

	memset(&msg, 0, sizeof(msg));
	(*seq)++;
	fill_header(&msg.hdr, ODL_SYNC_MSG_FILE_CHANGED,
		    sizeof(msg) - sizeof(msg.hdr), *seq);

	strncpy(msg.rel_path, rel_path, ODL_SYNC_PATH_MAX - 1);
	msg.rel_path[ODL_SYNC_PATH_MAX - 1] = '\0';
	msg.file_size = file_size;
	msg.mtime_ns  = mtime_ns;
	msg.mode      = mode;
	msg.is_dir    = is_dir ? 1 : 0;
	if (sha256)
		memcpy(msg.sha256, sha256, 32);

	return send_ctrl(h, sid, dst, &msg, sizeof(msg));
}

int odl_sync_send_file_removed(odl_tb5_t h, uint8_t sid, uint8_t dst,
			       uint32_t *seq,
			       const char *rel_path, bool is_dir)
{
	struct odl_sync_file_removed msg;

	memset(&msg, 0, sizeof(msg));
	(*seq)++;
	fill_header(&msg.hdr, ODL_SYNC_MSG_FILE_REMOVED,
		    sizeof(msg) - sizeof(msg.hdr), *seq);

	strncpy(msg.rel_path, rel_path, ODL_SYNC_PATH_MAX - 1);
	msg.rel_path[ODL_SYNC_PATH_MAX - 1] = '\0';
	msg.is_dir = is_dir ? 1 : 0;

	return send_ctrl(h, sid, dst, &msg, sizeof(msg));
}

int odl_sync_recv_msg(odl_tb5_t h, uint8_t sid, void *buf, size_t buf_size,
		      uint32_t *out_type)
{
	uint8_t src_id;
	uint32_t actual_len;
	struct odl_sync_header *hdr;
	size_t total_len;
	int ret;

	ret = odl_tb5_stream_wait_rx(h, sid, 2000);
	if (ret < 0)
		return ret;

	ret = odl_tb5_stream_recv(h, sid, buf, buf_size, &src_id, &actual_len);
	if (ret < 0)
		return ret;

	if (actual_len < sizeof(struct odl_sync_header))
		return -EPROTO;

	hdr = (struct odl_sync_header *)buf;

	if (hdr->magic != ODL_SYNC_MAGIC)
		return -EPROTO;

	if (hdr->version != ODL_SYNC_VERSION)
		return -EPROTO;

	/* Verify HMAC-SHA256 authentication tag in reserved field. */
	{
		uint8_t digest[32];
		unsigned int dlen = sizeof(digest);
		uint32_t expected_tag;

		HMAC(EVP_sha256(), ODL_SYNC_HMAC_KEY, sizeof(ODL_SYNC_HMAC_KEY) - 1,
		     (const uint8_t *)hdr, sizeof(*hdr) - sizeof(hdr->reserved),
		     digest, &dlen);
		memcpy(&expected_tag, digest, sizeof(expected_tag));
		if (hdr->reserved != expected_tag)
			return -EPROTO;
	}

	total_len = sizeof(struct odl_sync_header) + hdr->payload_len;
	if (total_len > actual_len)
		return -EPROTO;

	if (out_type)
		*out_type = hdr->type;

	return 0;
}

#define SHA256_READ_BUF_SIZE  (64 * 1024)

int odl_sync_sha256_file(const char *path, uint8_t out[32])
{
	FILE *fp = NULL;
	EVP_MD_CTX *ctx = NULL;
	unsigned char *read_buf = NULL;
	unsigned int digest_len = 0;
	size_t n;
	int ret = 0;

	fp = fopen(path, "rb");
	if (!fp)
		return -errno;

	ctx = EVP_MD_CTX_new();
	if (!ctx) {
		ret = -ENOMEM;
		goto out;
	}

	if (EVP_DigestInit_ex(ctx, EVP_sha256(), NULL) != 1) {
		ret = -EIO;
		goto out;
	}

	read_buf = malloc(SHA256_READ_BUF_SIZE);
	if (!read_buf) {
		ret = -ENOMEM;
		goto out;
	}

	while ((n = fread(read_buf, 1, SHA256_READ_BUF_SIZE, fp)) > 0) {
		if (EVP_DigestUpdate(ctx, read_buf, n) != 1) {
			ret = -EIO;
			goto out;
		}
	}

	if (ferror(fp)) {
		ret = -EIO;
		goto out;
	}

	if (EVP_DigestFinal_ex(ctx, out, &digest_len) != 1) {
		ret = -EIO;
		goto out;
	}

out:
	free(read_buf);
	if (ctx)
		EVP_MD_CTX_free(ctx);
	if (fp)
		fclose(fp);
	return ret;
}
