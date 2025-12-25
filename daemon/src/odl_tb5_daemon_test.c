/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2025-2026 OdinLink Project */
#include "odl_tb5_daemon_test.h"
#include "odl_tb5_daemon_dbus.h"
#include "odl_tb5_daemon_monitor.h"
#include "odl_tb5_daemon_sync.h"
#include "odl_tb5_cli.h"

#include "odl_tb5_daemon_sysinfo.h"

#include <glib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <pthread.h>

#define TEST_POOL_MAX_THREADS   2
#define PEER_WAIT_TIMEOUT_MS    5000
#define SERVER_PEER_TIMEOUT_MS  2000
#define STDOUT_CAPTURE_BUF_SZ   (256 * 1024)

struct odl_daemon_server_ctx {
	int           device_index;
	GThread      *thread;
	volatile int  running;
	pthread_t     tid;
	GAsyncQueue  *work_queue;
};

static GHashTable *test_table;
static GMutex      test_lock;
static GThreadPool *test_pool;

static struct odl_daemon_server_ctx server_ctx[ODL_DAEMON_MAX_DEVICES];
static GMutex server_lock;

static enum odl_cli_test_type test_type_from_str(const char *s)
{
	if (strcmp(s, "bandwidth") == 0)     return ODL_TEST_BANDWIDTH;
	if (strcmp(s, "latency") == 0)       return ODL_TEST_LATENCY;
	if (strcmp(s, "latency_load") == 0)  return ODL_TEST_LATENCY_LOAD;
	if (strcmp(s, "mimo") == 0)          return ODL_TEST_MIMO;
	if (strcmp(s, "jitter") == 0)        return ODL_TEST_JITTER;
	if (strcmp(s, "all") == 0)           return ODL_TEST_ALL;
	return (enum odl_cli_test_type)-1;
}

static const char *test_type_to_str(enum odl_cli_test_type t)
{
	switch (t) {
	case ODL_TEST_BANDWIDTH:    return "bandwidth";
	case ODL_TEST_LATENCY:      return "latency";
	case ODL_TEST_LATENCY_LOAD: return "latency_load";
	case ODL_TEST_MIMO:         return "mimo";
	case ODL_TEST_JITTER:       return "jitter";
	case ODL_TEST_ALL:          return "all";
	default:                    return "unknown";
	}
}

static void fill_default_params(struct odl_cli_params *p, int device_index,
				enum odl_cli_test_type test_type)
{
	memset(p, 0, sizeof(*p));
	p->device_index  = device_index;
	p->test_type     = test_type;
	p->block_sizes[0] = ODL_DEFAULT_BLOCK_SIZE;
	p->num_block_sizes = 1;
	p->iterations    = ODL_DEFAULT_ITERATIONS;
	p->duration_sec  = ODL_DEFAULT_DURATION;
	p->num_streams   = ODL_DEFAULT_STREAMS;
	p->bg_block_size = ODL_DEFAULT_BG_BLOCK_SIZE;
	p->warmup_iters  = ODL_DEFAULT_WARMUP;
	p->bidir         = false;
	p->verbose       = false;
	p->quiet         = true;
	p->output_file   = NULL;
}

struct stdout_capture {
	int  pipe_rd;
	int  pipe_wr;
	int  saved_fd;
};

static int stdout_capture_begin(struct stdout_capture *cap)
{
	int fds[2];

	cap->saved_fd = dup(STDOUT_FILENO);
	if (cap->saved_fd < 0)
		return -errno;

	if (pipe(fds) < 0) {
		int e = errno;
		close(cap->saved_fd);
		return -e;
	}

	cap->pipe_rd = fds[0];
	cap->pipe_wr = fds[1];

	fcntl(cap->pipe_rd, F_SETFL,
	      fcntl(cap->pipe_rd, F_GETFL) | O_NONBLOCK);

	if (dup2(cap->pipe_wr, STDOUT_FILENO) < 0) {
		int e = errno;
		close(cap->pipe_rd);
		close(cap->pipe_wr);
		dup2(cap->saved_fd, STDOUT_FILENO);
		close(cap->saved_fd);
		return -e;
	}

	return 0;
}

static char *stdout_capture_end(struct stdout_capture *cap)
{
	fflush(stdout);

	dup2(cap->saved_fd, STDOUT_FILENO);
	close(cap->saved_fd);

	close(cap->pipe_wr);

	char *buf = g_malloc(STDOUT_CAPTURE_BUF_SZ);
	size_t total = 0;
	ssize_t n;

	for (;;) {
		n = read(cap->pipe_rd, buf + total,
			 STDOUT_CAPTURE_BUF_SZ - total - 1);
		if (n <= 0)
			break;
		total += (size_t)n;
		if (total >= STDOUT_CAPTURE_BUF_SZ - 1)
			break;
	}
	buf[total] = '\0';

	close(cap->pipe_rd);
	return buf;
}

struct progress_idle_data {
	char    test_id[37];
	unsigned progress;
	char    subtest[64];
};

static gboolean emit_progress_idle(gpointer user_data)
{
	struct progress_idle_data *d = user_data;

	odl_daemon_dbus_emit_test_progress(d->test_id, d->progress,
					   d->subtest);
	g_free(d);
	return G_SOURCE_REMOVE;
}

static void post_progress(const struct odl_daemon_test_ctx *ctx,
			  unsigned progress, const char *subtest)
{
	struct progress_idle_data *d = g_new0(struct progress_idle_data, 1);

	g_strlcpy(d->test_id, ctx->uuid, sizeof(d->test_id));
	d->progress = progress;
	g_strlcpy(d->subtest, subtest ? subtest : "", sizeof(d->subtest));

	g_idle_add(emit_progress_idle, d);
}

struct completed_idle_data {
	char     test_id[37];
	gboolean success;
	char    *summary;
};

static gboolean emit_completed_idle(gpointer user_data)
{
	struct completed_idle_data *d = user_data;

	odl_daemon_dbus_emit_test_completed(d->test_id, d->success,
					    d->summary);
	g_free(d->summary);
	g_free(d);
	return G_SOURCE_REMOVE;
}

static void post_completed(const struct odl_daemon_test_ctx *ctx,
			   gboolean success, const char *summary)
{
	struct completed_idle_data *d = g_new0(struct completed_idle_data, 1);

	g_strlcpy(d->test_id, ctx->uuid, sizeof(d->test_id));
	d->success = success;
	d->summary = g_strdup(summary);

	g_idle_add(emit_completed_idle, d);
}

static char *build_result_json(const struct odl_daemon_test_ctx *ctx,
			       gboolean success)
{
	GString *js = g_string_new("{\n");

	g_string_append_printf(js, "  \"test_type\": \"%s\",\n",
			       ctx->test_type);
	g_string_append_printf(js, "  \"device_index\": %d,\n",
			       ctx->device_index);
	g_string_append_printf(js, "  \"success\": %s,\n",
			       success ? "true" : "false");

	if (ctx->output_text) {
		g_string_append(js, "  \"output\": \"");
		for (const char *p = ctx->output_text; *p; p++) {
			switch (*p) {
			case '"':  g_string_append(js, "\\\""); break;
			case '\\': g_string_append(js, "\\\\"); break;
			case '\n': g_string_append(js, "\\n");  break;
			case '\r': g_string_append(js, "\\r");  break;
			case '\t': g_string_append(js, "\\t");  break;
			default:
				if ((unsigned char)*p >= 0x20)
					g_string_append_c(js, *p);
				else
					g_string_append_printf(js, "\\u%04x",
							       (unsigned char)*p);
				break;
			}
		}
		g_string_append(js, "\"\n");
	} else {
		g_string_append(js, "  \"output\": null\n");
	}

	g_string_append(js, "}");

	return g_string_free(js, FALSE);
}

static int daemon_send_test_request(odl_tb5_t handle, uint8_t sid,
				    uint8_t dst,
				    const struct odl_cli_params *params,
				    enum odl_cli_test_type test_type,
				    uint32_t block_size)
{
	struct odl_cli_test_req req;
	char msg_buf[4096];
	uint32_t type, seq;
	int ret;

	memset(&req, 0, sizeof(req));
	req.test_type     = test_type;
	req.block_size    = block_size;
	req.iterations    = params->iterations;
	req.duration_sec  = params->duration_sec;
	req.num_streams   = params->num_streams;
	req.bg_block_size = params->bg_block_size;
	req.flags         = 0;
	if (params->bidir)
		req.flags |= ODL_TEST_FLAG_BIDIR;
	if (params->warmup_iters > 0)
		req.flags |= ODL_TEST_FLAG_WARMUP;

	ret = odl_cli_send_msg(handle, sid, dst, ODL_CLI_MSG_TEST_REQ, 0,
			       &req.test_type,
			       sizeof(req) - sizeof(req.hdr));
	if (ret < 0)
		return ret;

	ret = odl_cli_recv_msg(handle, sid, msg_buf, sizeof(msg_buf),
			       &type, &seq, NULL);
	if (ret < 0)
		return ret;

	if (type != ODL_CLI_MSG_TEST_ACK)
		return -EPROTO;

	return 0;
}

static int daemon_run_single_test(odl_tb5_t handle, uint8_t sid, uint8_t dst,
				  const struct odl_cli_params *params,
				  enum odl_cli_test_type test_type,
				  struct odl_daemon_test_ctx *ctx)
{
	const char *subtest = test_type_to_str(test_type);
	int ret;

	g_strlcpy(ctx->current_subtest, subtest, sizeof(ctx->current_subtest));
	post_progress(ctx, 0, subtest);

	switch (test_type) {
	case ODL_TEST_BANDWIDTH:
		for (int i = 0; i < params->num_block_sizes; i++) {
			ret = daemon_send_test_request(handle, sid, dst,
						       params, test_type,
						       params->block_sizes[i]);
			if (ret < 0)
				return ret;
			ret = odl_cli_bandwidth_client(handle, sid, dst,
						       params);
			if (ret < 0)
				return ret;
		}
		break;

	case ODL_TEST_LATENCY:
		ret = daemon_send_test_request(handle, sid, dst, params,
					       test_type,
					       params->block_sizes[0]);
		if (ret < 0)
			return ret;
		ret = odl_cli_latency_client(handle, sid, dst, params);
		break;

	case ODL_TEST_LATENCY_LOAD:
		ret = daemon_send_test_request(handle, sid, dst, params,
					       test_type,
					       params->block_sizes[0]);
		if (ret < 0)
			return ret;
		ret = odl_cli_latency_load_client(handle, sid, dst, params);
		break;

	case ODL_TEST_MIMO:
		ret = daemon_send_test_request(handle, sid, dst, params,
					       test_type,
					       params->block_sizes[0]);
		if (ret < 0)
			return ret;
		ret = odl_cli_mimo_client(handle, sid, dst, params);
		break;

	case ODL_TEST_JITTER:
		ret = daemon_send_test_request(handle, sid, dst, params,
					       test_type,
					       params->block_sizes[0]);
		if (ret < 0)
			return ret;
		ret = odl_cli_jitter_client(handle, sid, dst, params);
		break;

	default:
		return -EINVAL;
	}

	post_progress(ctx, 100, subtest);
	return ret;
}

static void test_worker(gpointer data, gpointer user_data)
{
	struct odl_daemon_test_ctx *ctx = data;
	(void)user_data;

	gboolean success = FALSE;
	int idx = ctx->device_index;

	ctx->state = ODL_DTEST_RUNNING;
	ctx->progress_pct = 0;
	post_progress(ctx, 0, ctx->test_type);

	g_mutex_lock(&server_lock);
	if (idx < 0 || idx >= ODL_DAEMON_MAX_DEVICES ||
	    !server_ctx[idx].thread || !server_ctx[idx].running) {
		g_mutex_unlock(&server_lock);
		g_printerr("test_worker[%s]: no device worker for %d\n",
			   ctx->uuid, idx);
		goto fail;
	}
	struct odl_daemon_server_ctx *sctx = &server_ctx[idx];
	g_mutex_unlock(&server_lock);

	struct odl_daemon_work_item work;
	memset(&work, 0, sizeof(work));
	work.type = ODL_WORK_TEST;
	work.test_ctx = ctx;
	work.done = FALSE;
	work.result = -1;
	g_mutex_init(&work.done_lock);
	g_cond_init(&work.done_cond);

	g_async_queue_push(sctx->work_queue, &work);

	pthread_kill(sctx->tid, SIGUSR1);

	g_mutex_lock(&work.done_lock);
	while (!work.done)
		g_cond_wait(&work.done_cond, &work.done_lock);
	g_mutex_unlock(&work.done_lock);

	g_mutex_clear(&work.done_lock);
	g_cond_clear(&work.done_cond);

	success = (work.result == 0);
	goto done;

fail:
	ctx->output_text = NULL;

done:
	if (success) {
		ctx->state = ODL_DTEST_COMPLETED;
		ctx->progress_pct = 100;
	} else if (ctx->state != ODL_DTEST_CANCELLED) {
		ctx->state = ODL_DTEST_FAILED;
	}

	ctx->result_json = build_result_json(ctx, success);

	post_completed(ctx, success,
		       ctx->result_json ? ctx->result_json : "{}");
}

static void sigusr1_nop(int sig)
{
	(void)sig;
}

static void signal_work_done(struct odl_daemon_work_item *w, int result)
{
	g_mutex_lock(&w->done_lock);
	w->result = result;
	w->done = TRUE;

	if (w->abandoned) {
		g_mutex_unlock(&w->done_lock);
		g_mutex_clear(&w->done_lock);
		g_cond_clear(&w->done_cond);
		g_free(w);
		return;
	}

	g_cond_signal(&w->done_cond);
	g_mutex_unlock(&w->done_lock);
}

static void drain_queue_not_connected(struct odl_daemon_server_ctx *sctx)
{
	struct odl_daemon_work_item *w;

	if (!sctx->work_queue)
		return;

	while ((w = g_async_queue_try_pop(sctx->work_queue)) != NULL)
		signal_work_done(w, -ENOTCONN);
}

static void handle_sysinfo_req(odl_tb5_t handle, uint8_t sid, uint8_t dst)
{
	struct odl_sysinfo si;
	struct odl_cli_sysinfo_payload payload;

	odl_daemon_sysinfo_collect(&si);

	memset(&payload, 0, sizeof(payload));
	payload.num_cpus = (uint32_t)si.num_cpus;
	payload.ram_total_mb = si.ram_total_mb;
	payload.ram_available_mb = si.ram_available_mb;
	payload.num_gpus = (uint32_t)si.num_gpus;

	for (int i = 0; i < si.num_cpus && i < ODL_SYSINFO_MAX_CPUS_WIRE; i++) {
		memcpy(payload.cpus[i].model, si.cpus[i].model,
		       sizeof(payload.cpus[i].model));
		payload.cpus[i].cores = si.cpus[i].cores;
		payload.cpus[i].threads = si.cpus[i].threads;
		payload.cpus[i].freq_mhz = si.cpus[i].freq_mhz;
	}

	for (int i = 0; i < si.num_gpus && i < ODL_SYSINFO_MAX_GPUS_WIRE; i++) {
		memcpy(payload.gpus[i].name, si.gpus[i].name,
		       sizeof(payload.gpus[i].name));
		payload.gpus[i].vram_total_mb = si.gpus[i].vram_total_mb;
		payload.gpus[i].vram_used_mb = si.gpus[i].vram_used_mb;
	}

	odl_cli_send_msg(handle, sid, dst, ODL_CLI_MSG_SYSINFO_RESP, 0,
			 &payload, sizeof(payload));
}

static void do_client_test(odl_tb5_t handle, uint8_t sid,
			   struct odl_daemon_work_item *work)
{
	struct odl_daemon_test_ctx *ctx = work->test_ctx;
	uint8_t dst = ODL_STREAM_TEST;
	struct stdout_capture cap;
	gboolean success = FALSE;
	int ret;

	ret = odl_cli_send_hello(handle, sid, dst);
	if (ret < 0) {
		g_printerr("do_client_test[%s]: send_hello failed: %s\n",
			   ctx->uuid, strerror(-ret));
		goto out;
	}

	ret = odl_cli_recv_hello(handle, sid);
	if (ret < 0) {
		g_printerr("do_client_test[%s]: recv_hello failed: %s\n",
			   ctx->uuid, strerror(-ret));
		goto out;
	}

	enum odl_cli_test_type ttype = test_type_from_str(ctx->test_type);
	struct odl_cli_params params;
	fill_default_params(&params, ctx->device_index, ttype);

	ret = stdout_capture_begin(&cap);
	if (ret < 0) {
		g_printerr("do_client_test[%s]: stdout capture failed: %s\n",
			   ctx->uuid, strerror(-ret));
		goto out;
	}

	if (ttype == ODL_TEST_ALL) {
		static const enum odl_cli_test_type all_tests[] = {
			ODL_TEST_BANDWIDTH,
			ODL_TEST_LATENCY,
			ODL_TEST_JITTER,
			ODL_TEST_LATENCY_LOAD,
			ODL_TEST_MIMO,
		};
		int ntests = (int)(sizeof(all_tests) / sizeof(all_tests[0]));

		for (int i = 0; i < ntests; i++) {
			if (ctx->state == ODL_DTEST_CANCELLED)
				break;

			struct odl_cli_params sub_params;
			fill_default_params(&sub_params, ctx->device_index,
					    all_tests[i]);

			unsigned pct = (unsigned)(i * 100 / ntests);
			ctx->progress_pct = (int)pct;
			post_progress(ctx, pct,
				      test_type_to_str(all_tests[i]));

			ret = daemon_run_single_test(handle, sid, dst,
						     &sub_params,
						     all_tests[i], ctx);
			if (ret < 0) {
				g_printerr("do_client_test[%s]: sub-test %s "
					   "failed: %s\n",
					   ctx->uuid,
					   test_type_to_str(all_tests[i]),
					   strerror(-ret));
				break;
			}
		}
	} else {
		ret = daemon_run_single_test(handle, sid, dst, &params,
					     ttype, ctx);
	}

	ctx->output_text = stdout_capture_end(&cap);

	odl_cli_send_msg(handle, sid, dst, ODL_CLI_MSG_DONE, 0, NULL, 0);

	if (ctx->state == ODL_DTEST_CANCELLED)
		success = FALSE;
	else if (ret >= 0)
		success = TRUE;

out:
	work->result = success ? 0 : -1;
}

static int parse_sysinfo_resp(const char *msg_buf, size_t msg_len,
			      struct odl_sysinfo *out)
{
	struct odl_cli_sysinfo_payload *p =
		(struct odl_cli_sysinfo_payload *)
		((const uint8_t *)msg_buf + sizeof(struct odl_cli_header));

	(void)msg_len;

	memset(out, 0, sizeof(*out));
	out->num_cpus = (int)p->num_cpus;
	if (out->num_cpus > ODL_SYSINFO_MAX_CPUS)
		out->num_cpus = ODL_SYSINFO_MAX_CPUS;
	out->ram_total_mb = p->ram_total_mb;
	out->ram_available_mb = p->ram_available_mb;
	out->num_gpus = (int)p->num_gpus;
	if (out->num_gpus > ODL_SYSINFO_MAX_GPUS)
		out->num_gpus = ODL_SYSINFO_MAX_GPUS;

	for (int i = 0; i < out->num_cpus; i++) {
		memcpy(out->cpus[i].model, p->cpus[i].model,
		       sizeof(out->cpus[i].model));
		out->cpus[i].model[sizeof(out->cpus[i].model) - 1] = '\0';
		out->cpus[i].cores = p->cpus[i].cores;
		out->cpus[i].threads = p->cpus[i].threads;
		out->cpus[i].freq_mhz = p->cpus[i].freq_mhz;
	}

	for (int i = 0; i < out->num_gpus; i++) {
		memcpy(out->gpus[i].name, p->gpus[i].name,
		       sizeof(out->gpus[i].name));
		out->gpus[i].name[sizeof(out->gpus[i].name) - 1] = '\0';
		out->gpus[i].vram_total_mb = p->gpus[i].vram_total_mb;
		out->gpus[i].vram_used_mb = p->gpus[i].vram_used_mb;
	}

	return 0;
}

static void do_sysinfo_exchange(odl_tb5_t handle, uint8_t sid,
				struct odl_daemon_work_item *work,
				volatile int *running)
{
	uint8_t dst = ODL_STREAM_TEST;
	char msg_buf[4096];
	uint32_t type, seq;
	uint8_t src_id;
	int ret;

	ret = odl_cli_send_msg(handle, sid, dst,
			       ODL_CLI_MSG_SYSINFO_REQ, 0, NULL, 0);
	if (ret < 0) {
		g_printerr("do_sysinfo_exchange: send SYSINFO_REQ failed: %s\n",
			   strerror(-ret));
		work->result = ret;
		return;
	}

	gint64 deadline = g_get_monotonic_time() + 10 * G_USEC_PER_SEC;

	for (;;) {
		if (running && !*running) {
			work->result = -EINTR;
			return;
		}
		if (g_get_monotonic_time() >= deadline) {
			g_printerr("do_sysinfo_exchange: "
				   "timed out (10s)\n");
			work->result = -ETIMEDOUT;
			return;
		}

		ret = odl_cli_recv_msg(handle, sid, msg_buf, sizeof(msg_buf),
				       &type, &seq, &src_id);
		if (ret == -EINTR || ret == -ETIMEDOUT)
			continue;
		if (ret < 0) {
			g_printerr("do_sysinfo_exchange: recv failed: %s\n",
				   strerror(-ret));
			work->result = ret;
			return;
		}

		if (type == ODL_CLI_MSG_SYSINFO_RESP) {
			work->result = parse_sysinfo_resp(msg_buf, (size_t)ret,
							  &work->sysinfo_result);
			return;
		}

		if (type == ODL_CLI_MSG_SYSINFO_REQ) {
			handle_sysinfo_req(handle, sid, src_id);
			continue;
		}

	}
}

static gpointer device_worker_thread(gpointer data)
{
	struct odl_daemon_server_ctx *sctx = data;
	int idx = sctx->device_index;

	sctx->tid = pthread_self();

	g_printerr("device_worker[%d]: starting\n", idx);

	while (sctx->running) {
		odl_tb5_t handle = NULL;
		char msg_buf[4096];
		uint32_t type, seq;
		uint8_t src_id;
		uint8_t sid = 0;
		int ret;

		ret = odl_tb5_open(&handle, idx);
		if (ret < 0) {
			if (sctx->running)
				g_printerr("device_worker[%d]: open "
					   "failed: %s (%d)\n",
					   idx, strerror(-ret), ret);
			drain_queue_not_connected(sctx);
			g_usleep(2 * G_USEC_PER_SEC);
			continue;
		}

		g_printerr("device_worker[%d]: device opened, "
			   "waiting for peer\n", idx);

		while (sctx->running) {
			ret = odl_tb5_wait_peer(handle,
						SERVER_PEER_TIMEOUT_MS);
			if (ret == 0)
				break;
			if (ret == -ETIMEDOUT || ret == -EINTR) {
				drain_queue_not_connected(sctx);
				continue;
			}
			g_printerr("device_worker[%d]: wait_peer "
				   "error: %s (%d)\n",
				   idx, strerror(-ret), ret);
			break;
		}

		if (ret < 0 || !sctx->running) {
			if (!sctx->running)
				g_printerr("device_worker[%d]: stop "
					   "requested during peer wait\n",
					   idx);
			odl_tb5_close(handle);
			if (ret < 0 && ret != -ETIMEDOUT && sctx->running)
				g_usleep(1 * G_USEC_PER_SEC);
			continue;
		}

		{
			struct odl_tb5_peer_info pinfo;
			bool ready = false;

			for (int w = 0; w < 80 && sctx->running; w++) {
				ret = odl_tb5_get_peer(handle, &pinfo);
				if (ret < 0)
					break;
				if (pinfo.state == ODL_TB5_STATE_READY) {
					ready = true;
					break;
				}
				if (pinfo.state != ODL_TB5_STATE_CONNECTED) {
					break;
				}
				g_usleep(100 * 1000);
			}

			if (!sctx->running) {
				g_printerr("device_worker[%d]: stop "
					   "requested during "
					   "READY wait\n", idx);
				odl_tb5_close(handle);
				continue;
			}

			if (!ready && pinfo.state != ODL_TB5_STATE_CONNECTED) {
				g_printerr("device_worker[%d]: peer state "
					   "went to %s, retrying\n", idx,
					   odl_daemon_state_str(
					     pinfo.state));
				odl_tb5_close(handle);
				g_usleep(2 * G_USEC_PER_SEC);
				continue;
			}

			if (ready) {
				g_printerr("device_worker[%d]: peer ready, "
					   "entering main loop\n", idx);
			} else {
				g_printerr("device_worker[%d]: DMA verify "
					   "timed out, proceeding in "
					   "CONNECTED state\n", idx);
			}
		}

		/* Open stream for this device worker */
		ret = odl_tb5_stream_open(handle, ODL_STREAM_TEST, &sid);
		if (ret < 0) {
			g_printerr("device_worker[%d]: stream_open "
				   "failed: %s (%d)\n",
				   idx, strerror(-ret), ret);
			odl_tb5_close(handle);
			g_usleep(2 * G_USEC_PER_SEC);
			continue;
		}

		while (sctx->running) {
			struct odl_daemon_work_item *work =
				g_async_queue_try_pop(sctx->work_queue);
			if (work) {
				const char *wtype =
					work->type == ODL_WORK_TEST ?
					"test" : "sysinfo";
				g_printerr("device_worker[%d]: processing "
					   "%s work item\n", idx, wtype);
				switch (work->type) {
				case ODL_WORK_TEST:
					do_client_test(handle, sid, work);
					break;
				case ODL_WORK_SYSINFO:
					do_sysinfo_exchange(handle, sid, work,
							    &sctx->running);
					break;
				default:
					break;
				}
				g_printerr("device_worker[%d]: %s work "
					   "done (result=%d)\n",
					   idx, wtype, work->result);
				signal_work_done(work, work->result);
				continue;
			}

			ret = odl_cli_recv_msg(handle, sid, msg_buf,
					       sizeof(msg_buf),
					       &type, &seq, &src_id);
			if (ret == -EINTR || ret == -ETIMEDOUT) {
				if (!sctx->running && ret == -EINTR)
					g_printerr("device_worker[%d]: "
						   "interrupted, stop "
						   "requested\n", idx);
				continue;
			}
			if (ret < 0) {
				g_printerr("device_worker[%d]: recv "
					   "error: %s (%d)\n",
					   idx, strerror(-ret), ret);
				break;
			}

			switch (type) {
			case ODL_CLI_MSG_HELLO:
				ret = odl_cli_send_msg(handle, sid, src_id,
						       ODL_CLI_MSG_HELLO_ACK,
						       0, NULL, 0);
				if (ret < 0) {
					g_printerr("device_worker[%d]: "
						   "HELLO_ACK failed\n",
						   idx);
					break;
				}

				g_printerr("device_worker[%d]: serving "
					   "remote test session\n", idx);

				while (sctx->running) {
					ret = odl_cli_recv_msg(
						handle, sid, msg_buf,
						sizeof(msg_buf),
						&type, &seq, &src_id);
					if (ret == -EINTR || ret == -ETIMEDOUT)
						continue;
					if (ret < 0)
						break;
					if (type == ODL_CLI_MSG_DONE) {
						g_printerr(
						  "device_worker[%d]: "
						  "client sent DONE\n",
						  idx);
						break;
					}
					if (type == ODL_CLI_MSG_SYSINFO_REQ) {
						handle_sysinfo_req(handle,
								   sid,
								   src_id);
						continue;
					}
					if (type != ODL_CLI_MSG_TEST_REQ)
						continue;

					struct odl_cli_test_req *req =
					  (struct odl_cli_test_req *)msg_buf;

					ret = odl_cli_send_msg(
						handle, sid, src_id,
						ODL_CLI_MSG_TEST_ACK,
						seq, NULL, 0);
					if (ret < 0)
						continue;

					switch (req->test_type) {
					case ODL_TEST_BANDWIDTH:
						odl_cli_bandwidth_server(
							handle, sid, src_id,
							req);
						break;
					case ODL_TEST_LATENCY:
						odl_cli_latency_server(
							handle, sid, src_id,
							req);
						break;
					case ODL_TEST_LATENCY_LOAD:
						odl_cli_latency_load_server(
							handle, sid, src_id,
							req);
						break;
					case ODL_TEST_MIMO:
						odl_cli_mimo_server(
							handle, sid, src_id,
							req);
						break;
					case ODL_TEST_JITTER:
						odl_cli_jitter_server(
							handle, sid, src_id,
							req);
						break;
					default:
						break;
					}
				}

				g_printerr("device_worker[%d]: session "
					   "ended\n", idx);
				break;

			case ODL_CLI_MSG_SYSINFO_REQ:
				handle_sysinfo_req(handle, sid, src_id);
				break;

			default:
				break;
			}
		}

		drain_queue_not_connected(sctx);
		odl_tb5_stream_close(handle, sid);
		odl_tb5_close(handle);

		if (sctx->running) {
			g_printerr("device_worker[%d]: connection lost, "
				   "reconnecting in 2s\n", idx);
			g_usleep(2 * G_USEC_PER_SEC);
		} else {
			g_printerr("device_worker[%d]: device removed, "
				   "exiting\n", idx);
		}
	}

	drain_queue_not_connected(sctx);
	g_printerr("device_worker[%d]: stopped (running=%d)\n",
		   idx, sctx->running);
	return NULL;
}

void odl_daemon_server_start_for_device(int index)
{
	g_mutex_lock(&server_lock);

	if (index < 0 || index >= ODL_DAEMON_MAX_DEVICES)
		goto out;

	if (server_ctx[index].thread) {
		goto out;
	}

	if (odl_daemon_sync_owns_device(index)) {
		g_printerr("server_start: skipping device %d "
			   "(owned by sync engine)\n", index);
		goto out;
	}

	server_ctx[index].device_index = index;
	server_ctx[index].running = 1;

	char name[32];
	g_snprintf(name, sizeof(name), "dev-%d", index);
	server_ctx[index].work_queue = g_async_queue_new();
	server_ctx[index].thread = g_thread_new(name,
						device_worker_thread,
						&server_ctx[index]);

out:
	g_mutex_unlock(&server_lock);
}

void odl_daemon_server_stop_for_device(int index)
{
	g_mutex_lock(&server_lock);

	if (index < 0 || index >= ODL_DAEMON_MAX_DEVICES)
		goto out;

	if (!server_ctx[index].thread)
		goto out;

	server_ctx[index].running = 0;

	pthread_t tid = server_ctx[index].tid;
	GThread *thread = server_ctx[index].thread;

	g_mutex_unlock(&server_lock);

	for (int i = 0; i < 3; i++) {
		if (tid)
			pthread_kill(tid, SIGUSR1);
		g_usleep(50 * 1000);
	}

	g_thread_join(thread);

	g_mutex_lock(&server_lock);
	server_ctx[index].thread = NULL;
	server_ctx[index].tid = 0;

	if (server_ctx[index].work_queue) {
		g_async_queue_unref(server_ctx[index].work_queue);
		server_ctx[index].work_queue = NULL;
	}

out:
	g_mutex_unlock(&server_lock);
}

static void server_start_all(void)
{
	g_mutex_lock(&g_device_table.lock);

	for (int i = 0; i < ODL_DAEMON_MAX_DEVICES; i++) {
		if (g_device_table.slots[i].present)
			odl_daemon_server_start_for_device(i);
	}

	g_mutex_unlock(&g_device_table.lock);
}

static void server_stop_all(void)
{
	for (int i = 0; i < ODL_DAEMON_MAX_DEVICES; i++)
		odl_daemon_server_stop_for_device(i);
}

static void test_ctx_free(gpointer data)
{
	struct odl_daemon_test_ctx *ctx = data;

	g_free(ctx->result_json);
	g_free(ctx->output_text);
	g_free(ctx);
}

int odl_daemon_test_init(void)
{
	GError *err = NULL;

	g_mutex_init(&test_lock);
	g_mutex_init(&server_lock);

	test_table = g_hash_table_new_full(g_str_hash, g_str_equal,
					   NULL, test_ctx_free);

	test_pool = g_thread_pool_new(test_worker, NULL,
				      TEST_POOL_MAX_THREADS,
				      FALSE, &err);
	if (!test_pool) {
		g_printerr("odl_daemon_test_init: pool creation failed: %s\n",
			   err ? err->message : "unknown");
		if (err)
			g_error_free(err);
		return -1;
	}

	struct sigaction sa;
	memset(&sa, 0, sizeof(sa));
	sa.sa_handler = sigusr1_nop;
	sigemptyset(&sa.sa_mask);
	sa.sa_flags = 0;
	sigaction(SIGUSR1, &sa, NULL);

	g_printerr("odl_daemon_test_init: ready (pool=%d threads)\n",
		   TEST_POOL_MAX_THREADS);
	return 0;
}

void odl_daemon_test_shutdown(void)
{
	server_stop_all();

	if (test_pool) {
		g_thread_pool_free(test_pool, FALSE, TRUE);
		test_pool = NULL;
	}

	g_mutex_lock(&test_lock);
	if (test_table) {
		g_hash_table_destroy(test_table);
		test_table = NULL;
	}
	g_mutex_unlock(&test_lock);

	g_mutex_clear(&test_lock);
	g_mutex_clear(&server_lock);

	g_printerr("odl_daemon_test_shutdown: done\n");
}

const char *odl_daemon_test_run(int device_index, const char *test_type)
{
	if (!test_type || !test_pool)
		return NULL;

	enum odl_cli_test_type ttype = test_type_from_str(test_type);
	if ((int)ttype < 0)
		return NULL;

	if (device_index < 0 || device_index >= ODL_DAEMON_MAX_DEVICES)
		return NULL;

	struct odl_daemon_test_ctx *ctx =
		g_new0(struct odl_daemon_test_ctx, 1);

	gchar *uuid = g_uuid_string_random();
	g_strlcpy(ctx->uuid, uuid, sizeof(ctx->uuid));
	g_free(uuid);

	ctx->device_index = device_index;
	g_strlcpy(ctx->test_type, test_type, sizeof(ctx->test_type));
	ctx->state = ODL_DTEST_QUEUED;
	ctx->progress_pct = 0;

	g_mutex_lock(&test_lock);
	g_hash_table_insert(test_table, ctx->uuid, ctx);
	g_mutex_unlock(&test_lock);

	GError *err = NULL;
	if (!g_thread_pool_push(test_pool, ctx, &err)) {
		g_printerr("test_run: pool push failed: %s\n",
			   err ? err->message : "unknown");
		if (err)
			g_error_free(err);

		g_mutex_lock(&test_lock);
		g_hash_table_remove(test_table, ctx->uuid);
		g_mutex_unlock(&test_lock);
		return NULL;
	}

	g_printerr("test_run: queued %s test on device %d -> %s\n",
		   test_type, device_index, ctx->uuid);

	return ctx->uuid;
}

bool odl_daemon_test_cancel(const char *test_id)
{
	if (!test_id)
		return false;

	g_mutex_lock(&test_lock);
	struct odl_daemon_test_ctx *ctx =
		g_hash_table_lookup(test_table, test_id);
	g_mutex_unlock(&test_lock);

	if (!ctx)
		return false;

	if (ctx->state == ODL_DTEST_QUEUED ||
	    ctx->state == ODL_DTEST_RUNNING) {
		ctx->state = ODL_DTEST_CANCELLED;
		g_printerr("test_cancel: %s -> cancelled\n", test_id);
		return true;
	}

	return false;
}

struct odl_daemon_test_ctx *odl_daemon_test_find(const char *test_id)
{
	if (!test_id)
		return NULL;

	g_mutex_lock(&test_lock);
	struct odl_daemon_test_ctx *ctx =
		g_hash_table_lookup(test_table, test_id);
	g_mutex_unlock(&test_lock);

	return ctx;
}

int odl_daemon_test_request_peer_sysinfo(int device_index,
					  struct odl_sysinfo *out)
{
	if (!out || device_index < 0 || device_index >= ODL_DAEMON_MAX_DEVICES)
		return -EINVAL;

	if (odl_daemon_sync_owns_device(device_index))
		return odl_daemon_sync_request_sysinfo(device_index, out);

	g_mutex_lock(&server_lock);
	if (!server_ctx[device_index].thread ||
	    !server_ctx[device_index].running) {
		g_mutex_unlock(&server_lock);
		return -ENODEV;
	}
	struct odl_daemon_server_ctx *sctx = &server_ctx[device_index];
	g_mutex_unlock(&server_lock);

	struct odl_daemon_work_item *work = g_new0(struct odl_daemon_work_item, 1);
	work->type = ODL_WORK_SYSINFO;
	work->done = FALSE;
	work->result = -1;
	work->abandoned = FALSE;
	g_mutex_init(&work->done_lock);
	g_cond_init(&work->done_cond);

	g_async_queue_push(sctx->work_queue, work);
	pthread_kill(sctx->tid, SIGUSR1);

	g_mutex_lock(&work->done_lock);
	gint64 end_time = g_get_monotonic_time() + 10 * G_USEC_PER_SEC;
	while (!work->done) {
		if (!g_cond_wait_until(&work->done_cond, &work->done_lock,
				       end_time)) {
			work->abandoned = TRUE;
			g_mutex_unlock(&work->done_lock);
			return -ETIMEDOUT;
		}
	}
	g_mutex_unlock(&work->done_lock);

	int ret = work->result;
	if (ret == 0)
		memcpy(out, &work->sysinfo_result, sizeof(*out));

	g_mutex_clear(&work->done_lock);
	g_cond_clear(&work->done_cond);
	g_free(work);

	return ret;
}
