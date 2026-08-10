/*
 * OdinLink — CLI: Entry Point (Parses Arguments, Dispatches)
 *
 * Decides whether to run as server (wait for peer) or client (connect
 * and run a test). Dispatches to bandwidth/latency/jitter/MIMO tests.
 */
#include "odl_tb5_cli.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

static void print_usage(const char *prog)
{
	fprintf(stderr,
		"OdinLink TB5 Test CLI\n"
		"\n"
		"Usage: %s <mode> [options]\n"
		"\n"
		"Modes:\n"
		"  server          Wait for peer and respond to tests\n"
		"  client          Connect to peer and run tests\n"
		"\n"
		"Common options:\n"
		"  -d <N>          Device index (default: 0)\n"
		"  -v              Verbose output\n"
		"  -q              Quiet (machine-readable)\n"
		"  -o <file>       Output results to CSV file\n"
		"\n"
		"Client options:\n"
		"  -t <test>       Test type: bandwidth|latency|latency-load|mimo|jitter|all\n"
		"  -b <sizes>      Block sizes, comma-separated (e.g., 4K,64K,1M,4M)\n"
		"  -i <count>      Iteration count (default: 1000 for latency, 100 for bw)\n"
		"  -D <seconds>    Duration per test (default: 10)\n"
		"  --bidir         Bidirectional test (bandwidth)\n"
		"  --streams <N>   Parallel streams (mimo, default: 4)\n"
		"  --bg-size <sz>  Background block size (latency-load, default: 1M)\n"
		"  --warmup <N>    Warmup iterations (default: 10)\n",
		prog);
}

static uint32_t parse_size(const char *str)
{
	char *end;
	unsigned long val = strtoul(str, &end, 10);

	switch (*end) {
	case 'K': case 'k': val *= 1024; break;
	case 'M': case 'm': val *= 1024 * 1024; break;
	case 'G': case 'g': val *= 1024UL * 1024 * 1024; break;
	}
	return (uint32_t)val;
}

static void parse_block_sizes(const char *str, struct odl_cli_params *p)
{
	char buf[256];
	char *tok, *saveptr;

	snprintf(buf, sizeof(buf), "%s", str);
	p->num_block_sizes = 0;

	/* Accept comma or slash separators (e.g. 4K,64K,1M or 4K/64K/1M). */
	tok = strtok_r(buf, ",/", &saveptr);
	while (tok && p->num_block_sizes < 16) {
		p->block_sizes[p->num_block_sizes++] = parse_size(tok);
		tok = strtok_r(NULL, ",/", &saveptr);
	}
}

static enum odl_cli_test_type parse_test_type(const char *str)
{
	if (strcmp(str, "bandwidth") == 0 || strcmp(str, "bw") == 0)
		return ODL_TEST_BANDWIDTH;
	if (strcmp(str, "latency") == 0 || strcmp(str, "lat") == 0)
		return ODL_TEST_LATENCY;
	if (strcmp(str, "latency-load") == 0 || strcmp(str, "lat-load") == 0)
		return ODL_TEST_LATENCY_LOAD;
	if (strcmp(str, "mimo") == 0)
		return ODL_TEST_MIMO;
	if (strcmp(str, "jitter") == 0)
		return ODL_TEST_JITTER;
	if (strcmp(str, "all") == 0)
		return ODL_TEST_ALL;

	fprintf(stderr, "Unknown test type: %s\n", str);
	return 0;
}

int main(int argc, char *argv[])
{
	struct odl_cli_params params = {
		.device_index = 0,
		.test_type = 0,
		.num_block_sizes = 0,
		.iterations = 0,
		.duration_sec = ODL_DEFAULT_DURATION,
		.num_streams = ODL_DEFAULT_STREAMS,
		.bg_block_size = ODL_DEFAULT_BG_BLOCK_SIZE,
		.warmup_iters = ODL_DEFAULT_WARMUP,
		.bidir = false,
		.verbose = false,
		.quiet = false,
		.output_file = NULL,
	};

	static struct option long_opts[] = {
		{ "bidir",   no_argument,       NULL, 'B' },
		{ "streams", required_argument, NULL, 'S' },
		{ "bg-size", required_argument, NULL, 'G' },
		{ "warmup",  required_argument, NULL, 'W' },
		{ "help",    no_argument,       NULL, 'h' },
		{ NULL, 0, NULL, 0 }
	};

	if (argc < 2) {
		print_usage(argv[0]);
		return 1;
	}

	const char *mode = argv[1];
	bool is_server = (strcmp(mode, "server") == 0);
	bool is_client = (strcmp(mode, "client") == 0);

	if (!is_server && !is_client) {
		fprintf(stderr, "Unknown mode: %s\n", mode);
		print_usage(argv[0]);
		return 1;
	}

	optind = 2;
	int opt;
	while ((opt = getopt_long(argc, argv, "d:t:b:i:D:o:vqh",
				  long_opts, NULL)) != -1) {
		switch (opt) {
		case 'd':
			params.device_index = atoi(optarg);
			break;
		case 't':
			params.test_type = parse_test_type(optarg);
			if (!params.test_type)
				return 1;
			break;
		case 'b':
			parse_block_sizes(optarg, &params);
			break;
		case 'i':
			params.iterations = (uint32_t)atoi(optarg);
			break;
		case 'D':
			params.duration_sec = (uint32_t)atoi(optarg);
			break;
		case 'o':
			params.output_file = optarg;
			break;
		case 'v':
			params.verbose = true;
			break;
		case 'q':
			params.quiet = true;
			break;
		case 'B':
			params.bidir = true;
			break;
		case 'S':
			params.num_streams = (uint32_t)atoi(optarg);
			break;
		case 'G':
			params.bg_block_size = parse_size(optarg);
			break;
		case 'W':
			params.warmup_iters = (uint32_t)atoi(optarg);
			break;
		case 'h':
			print_usage(argv[0]);
			return 0;
		default:
			print_usage(argv[0]);
			return 1;
		}
	}

	if (params.num_block_sizes == 0) {
		params.block_sizes[0] = ODL_DEFAULT_BLOCK_SIZE;
		params.num_block_sizes = 1;
	}

	if (params.iterations == 0) {
		switch (params.test_type) {
		case ODL_TEST_LATENCY:
		case ODL_TEST_JITTER:
			params.iterations = 10000;
			break;
		case ODL_TEST_LATENCY_LOAD:
			params.iterations = 5000;
			break;
		default:
			params.iterations = ODL_DEFAULT_ITERATIONS;
			break;
		}
	}

	if (is_client && !params.test_type) {
		fprintf(stderr, "Client mode requires -t <test>\n");
		return 1;
	}

	if (is_server)
		return odl_cli_run_server(&params);
	else
		return odl_cli_run_client(&params);
}
