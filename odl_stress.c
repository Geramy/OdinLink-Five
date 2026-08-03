/* odl_stress.c — reproduce the RCCL plugin's framing under concurrency, WITHOUT
 * RCCL/vLLM, to isolate whether the OdinLink stream layer loses/desyncs messages.
 *
 * Faithfully mirrors plugin do_transfer(): each message = [4-byte size header]
 * + payload in <=ODL_CHUNK single-frame sends, over N concurrent streams that
 * share ONE handle (like the plugin's g_handle). Payload carries a 4-byte
 * sequence number so the receiver detects lost/reordered/short/empty messages.
 *
 *   server:  ./odl_stress server <N> <msg_bytes> <iters>
 *   client:  ./odl_stress client <N> <msg_bytes> <iters>
 * Server opens recv streams with FIXED ids 1..N; client sends to dst=1..N.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <pthread.h>
#include <odl_tb5/odl_tb5.h>

#define ODL_CHUNK 4000

static odl_tb5_t H;
static int N, ITERS;
static size_t MSG;

struct sctx { int idx; uint8_t sid; uint8_t dst; };

/* Mixed message sizes like RCCL: tiny control-ish + medium + large, so the
 * kernel rx demux sees small frames interleaved with big chunked transfers. */
static size_t size_for(int it)
{
	switch (it & 3) {
	case 0:  return MSG;      /* large  (1 MB) */
	case 1:  return 350208;   /* medium */
	case 2:  return 128;      /* tiny   */
	default: return 8192;     /* small  */
	}
}

static void fill(uint8_t *b, size_t n, uint32_t seq)
{
	memcpy(b, &seq, 4);
	for (size_t j = 4; j < n; j++) b[j] = (uint8_t)((seq + j) & 0xff);
}

/* ---- client: one sender thread per stream (plugin send framing) ---- */
static void *sender(void *a)
{
	struct sctx *c = a;
	uint8_t *buf = malloc(MSG);
	for (int it = 0; it < ITERS; it++) {
		size_t sz = size_for(it);
		fill(buf, sz, (uint32_t)it);
		uint32_t hdr = (uint32_t)sz;
		int ret = odl_tb5_stream_send(H, c->sid, c->dst, &hdr, sizeof(hdr));
		size_t off = 0;
		while (ret >= 0 && off < sz) {
			size_t n = sz - off; if (n > ODL_CHUNK) n = ODL_CHUNK;
			ret = odl_tb5_stream_send(H, c->sid, c->dst, buf + off, (uint32_t)n);
			off += n;
		}
		if (ret < 0) { printf("[s%d] SEND FAIL it=%d ret=%d\n", c->idx, it, ret); break; }
	}
	printf("[s%d] sender done (%d msgs to dst=%u)\n", c->idx, ITERS, c->dst);
	free(buf); return NULL;
}

/* ---- server: one receiver thread per stream (plugin recv framing) ---- */
static void *receiver(void *a)
{
	struct sctx *c = a;
	uint8_t *buf = malloc(MSG);
	long ok = 0, zero = 0, shorthdr = 0, shortpay = 0, seqbad = 0, bytebad = 0;
	for (int expect = 0; expect < ITERS; expect++) {
		size_t expsz = size_for(expect);
		uint32_t total = 0, actual = 0; uint8_t src = 0;
		int ret = odl_tb5_stream_recv(H, c->sid, &total, sizeof(total), &src, &actual);
		if (ret < 0) { printf("[r%d] RECV hdr FAIL expect=%d ret=%d\n", c->idx, expect, ret); break; }
		if (actual != sizeof(total)) { shorthdr++; printf("[r%d] SHORT HDR expect=%d actual=%u\n", c->idx, expect, actual); continue; }
		if (total != expsz) { zero += (total == 0); seqbad += (total != 0); printf("[r%d] HDR MISMATCH expect=%d want_sz=%zu got_sz=%u (DESYNC)\n", c->idx, expect, expsz, total); }
		if (total > MSG) total = MSG;
		size_t off = 0; int bad = 0;
		while (off < total) {
			size_t n = total - off; if (n > ODL_CHUNK) n = ODL_CHUNK;
			ret = odl_tb5_stream_recv(H, c->sid, buf + off, (uint32_t)n, &src, &actual);
			if (ret < 0) { printf("[r%d] RECV chunk FAIL expect=%d ret=%d\n", c->idx, expect, ret); bad = 1; break; }
			off += actual;
			if (actual == 0) break;
		}
		if (off != expsz) { shortpay++; if (shortpay<=3) printf("[r%d] SHORT PAYLOAD expect=%d got=%zu/%zu\n", c->idx, expect, off, expsz); continue; }
		uint32_t seq; memcpy(&seq, buf, 4);
		if ((int)seq != expect) { seqbad++; if (seqbad<=3) printf("[r%d] SEQ MISMATCH want=%d got=%u (desync!)\n", c->idx, expect, seq); }
		else if (buf[expsz/2] != (uint8_t)((seq + expsz/2) & 0xff)) { bytebad++; }
		else ok++;
		(void)bad;
		if ((expect % 50) == 0) printf("[r%d] progress %d/%d ok=%ld\n", c->idx, expect, ITERS, ok);
	}
	printf("[r%d] SUMMARY sid=%u ok=%ld zero=%ld shorthdr=%ld shortpay=%ld seqbad=%ld bytebad=%ld\n",
	       c->idx, c->sid, ok, zero, shorthdr, shortpay, seqbad, bytebad);
	free(buf); return NULL;
}

int main(int argc, char **argv)
{
	if (argc < 5) { fprintf(stderr, "usage: %s server|client N msg iters\n", argv[0]); return 2; }
	int server = !strcmp(argv[1], "server");
	N = atoi(argv[2]); MSG = strtoul(argv[3], NULL, 10); ITERS = atoi(argv[4]);
	setvbuf(stdout, NULL, _IOLBF, 0);

	if (odl_tb5_open(&H, 0) < 0) { perror("open"); return 1; }
	if (odl_tb5_wait_peer(H, 15000) < 0) { fprintf(stderr, "wait_peer failed\n"); return 1; }
	printf("%s: link up, N=%d msg=%zu iters=%d chunk=%d\n", argv[1], N, MSG, ITERS, ODL_CHUNK);

	int duplex = !strcmp(argv[1], "duplex");
	if (duplex) {
		/* Both nodes send AND receive on N streams each — mirrors vLLM,
		 * where every rank drives TX and RX on the shared handle at once
		 * (TX/RX contend for the same frame pool). Recv ids fixed 1..N;
		 * senders target the peer's ids 1..N. */
		struct sctx *rc = calloc(N, sizeof(*rc)), *sc = calloc(N, sizeof(*sc));
		pthread_t *rt = calloc(N, sizeof(*rt)), *st = calloc(N, sizeof(*st));
		for (int i = 0; i < N; i++) {
			uint8_t rsid, ssid;
			if (odl_tb5_stream_open(H, (uint8_t)(i + 1), &rsid) < 0) { fprintf(stderr, "recv stream_open %d failed\n", i+1); return 1; }
			if (odl_tb5_stream_open(H, 0, &ssid) < 0) { fprintf(stderr, "send stream_open failed\n"); return 1; }
			rc[i].idx = i; rc[i].sid = rsid;
			sc[i].idx = i; sc[i].sid = ssid; sc[i].dst = (uint8_t)(i + 1);
		}
		printf("duplex: %d recv + %d send streams open\n", N, N);
		sleep(3);  /* both post recv streams before anyone sends */
		for (int i = 0; i < N; i++) pthread_create(&rt[i], NULL, receiver, &rc[i]);
		for (int i = 0; i < N; i++) pthread_create(&st[i], NULL, sender,   &sc[i]);
		for (int i = 0; i < N; i++) pthread_join(st[i], NULL);
		for (int i = 0; i < N; i++) pthread_join(rt[i], NULL);
		printf("duplex: ALL THREADS JOINED\n");
		odl_tb5_close(H);
		return 0;
	}

	struct sctx *c = calloc(N, sizeof(*c));
	pthread_t *t = calloc(N, sizeof(*t));
	for (int i = 0; i < N; i++) {
		c[i].idx = i; c[i].dst = (uint8_t)(i + 1);
		uint8_t sid;
		if (server) {
			/* fixed recv id i+1 so client can target it */
			if (odl_tb5_stream_open(H, (uint8_t)(i + 1), &sid) < 0) { fprintf(stderr, "srv stream_open %d failed\n", i+1); return 1; }
		} else {
			if (odl_tb5_stream_open(H, 0, &sid) < 0) { fprintf(stderr, "cli stream_open failed\n"); return 1; }
		}
		c[i].sid = sid;
	}
	printf("%s: %d streams open\n", argv[1], N);
	if (!server) sleep(2);  /* let server post its recv streams first */

	for (int i = 0; i < N; i++)
		pthread_create(&t[i], NULL, server ? receiver : sender, &c[i]);
	for (int i = 0; i < N; i++)
		pthread_join(t[i], NULL);

	printf("%s: ALL THREADS JOINED\n", argv[1]);
	odl_tb5_close(H);
	return 0;
}
