/*
 * OdinLink — CLI: Full Link Diagnosis ("diag" mode)
 *
 * Walks every layer between the Thunderbolt controller and a READY
 * OdinLink device and reports, in plain English, the first one that is
 * broken:
 *
 *   controller → cable detected → link trained → peer discovered
 *   (XDomain) → OdinLink driver bound → login handshake → READY
 *
 * Reads sysfs, and (as root) the thunderbolt and odl_tb5 debugfs. Works
 * on any kernel; degrades gracefully when debugfs is unavailable.
 */
#include "odl_tb5_cli.h"
#include "odl_tb5_uapi.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>

#define TB_SYS   "/sys/bus/thunderbolt/devices"
#define TB_DBG   "/sys/kernel/debug/thunderbolt"
#define ODL_DBG  "/sys/kernel/debug/odl_tb5/status"

/* Lane adapter states from LANE_ADP_CS_1 (USB4 spec) */
enum lane_state {
	LANE_DISABLED  = 0,
	LANE_TRAINING  = 1,
	LANE_UP        = 2,	/* CL0; 3-6 are CLx power states, also up */
	LANE_UNPLUGGED = 7,
	LANE_UNKNOWN   = -1,
};

static const char *lane_state_name(int st)
{
	switch (st) {
	case LANE_DISABLED:  return "disabled";
	case LANE_TRAINING:  return "training";
	case 2: case 3: case 4: case 5: case 6:
			     return "up";
	case LANE_UNPLUGGED: return "unplugged";
	default:             return "unknown";
	}
}

static const char *lane_speed_name(unsigned int speed_bits)
{
	switch (speed_bits) {
	case 0x8: return "10 Gb/s (Gen 2)";
	case 0x4: return "20 Gb/s (Gen 3)";
	case 0x2: return "40 Gb/s (Gen 4)";
	default:  return "?";
	}
}

static bool file_exists(const char *path)
{
	struct stat st;
	return stat(path, &st) == 0;
}

static int read_line(const char *path, char *buf, size_t len)
{
	FILE *f = fopen(path, "r");

	if (!f)
		return -1;
	if (!fgets(buf, len, f)) {
		fclose(f);
		return -1;
	}
	fclose(f);
	buf[strcspn(buf, "\n")] = '\0';
	return 0;
}

/* Read LANE_ADP_CS_1 (dword offset 0x37) from a port's debugfs regs dump.
 * Returns the raw register value, or 0 if unreadable. */
static unsigned long read_lane_adp_cs_1(const char *router, int port)
{
	char path[256], line[128];
	unsigned long val = 0;
	FILE *f;

	snprintf(path, sizeof(path), TB_DBG "/%s/port%d/regs", router, port);
	f = fopen(path, "r");
	if (!f)
		return 0;
	while (fgets(line, sizeof(line), f)) {
		unsigned long off, rel, cap, vs, v;

		if (sscanf(line, "%lx %lx %lx %lx %lx",
			   &off, &rel, &cap, &vs, &v) == 5 && off == 0x37) {
			val = v;
			break;
		}
	}
	fclose(f);
	return val;
}

struct lane_summary {
	int best_state;		/* most-alive state seen on any lane */
	int flapping;		/* state changed between two samples */
	char router[32];
	int port;
	unsigned long reg;
};

static int state_rank(int st)
{
	if (st >= LANE_UP && st <= 6)
		return 3;
	if (st == LANE_TRAINING)
		return 2;
	if (st == LANE_UNPLUGGED)
		return 1;
	return 0;
}

/* Scan every host router's lane adapters. Two passes, ~1.5 s apart, so a
 * link that cycles train -> drop -> train shows up as "flapping". */
static void scan_lanes(struct lane_summary *sum, bool verbose)
{
	struct dirent *de;
	DIR *d;
	int pass;

	sum->best_state = LANE_UNKNOWN;
	sum->flapping = 0;

	for (pass = 0; pass < 2; pass++) {
		d = opendir(TB_SYS);
		if (!d)
			return;
		while ((de = readdir(d))) {
			char path[512];
			int dom, port;

			if (sscanf(de->d_name, "%d-0", &dom) != 1 ||
			    strchr(de->d_name, '.'))
				continue;
			if (strcmp(de->d_name + strlen(de->d_name) - 2, "-0"))
				continue;

			for (port = 1; port < 8; port++) {
				unsigned long reg;
				int st;

				snprintf(path, sizeof(path),
					 TB_DBG "/%s/port%d/regs",
					 de->d_name, port);
				if (!file_exists(path))
					continue;
				reg = read_lane_adp_cs_1(de->d_name, port);
				if (!reg)
					continue;
				st = (reg >> 26) & 0xf;
				if (pass == 0 && verbose)
					printf("    %s port%d: %s (raw 0x%08lx)\n",
					       de->d_name, port,
					       lane_state_name(st), reg);
				if (pass == 1 && sum->port == port &&
				    !strcmp(sum->router, de->d_name)) {
					int prev = (sum->reg >> 26) & 0xf;

					if (prev != st)
						sum->flapping = 1;
				}
				if (state_rank(st) > state_rank(sum->best_state) ||
				    sum->best_state == LANE_UNKNOWN) {
					sum->best_state = st;
					sum->port = port;
					sum->reg = reg;
					snprintf(sum->router,
						 sizeof(sum->router), "%s",
						 de->d_name);
				}
			}
		}
		closedir(d);
		if (pass == 0 && sum->best_state == LANE_TRAINING)
			usleep(1500 * 1000);
		else
			break;
	}
}

/* Fallback when debugfs is unavailable: sysfs "link" attribute only says
 * up (usb4/tbt) or nothing. */
static int scan_lanes_sysfs(void)
{
	struct dirent *de;
	DIR *d = opendir(TB_SYS);
	int up = 0;

	if (!d)
		return 0;
	while ((de = readdir(d))) {
		char path[512], buf[32];
		int port;

		if (strchr(de->d_name, '.') || !strstr(de->d_name, "-0"))
			continue;
		for (port = 1; port < 8; port++) {
			snprintf(path, sizeof(path),
				 TB_SYS "/%s/usb4_port%d/link",
				 de->d_name, port);
			if (read_line(path, buf, sizeof(buf)) == 0 &&
			    strcmp(buf, "none") && buf[0])
				up++;
		}
	}
	closedir(d);
	return up;
}

/* Count XDomain entries (peer hosts) on the thunderbolt bus. Routers are
 * "<domain>-0", services contain '.', everything else with a '-' is a
 * discovered peer. */
static int count_xdomains(void)
{
	struct dirent *de;
	DIR *d = opendir(TB_SYS);
	int n = 0;

	if (!d)
		return 0;
	while ((de = readdir(d))) {
		size_t len = strlen(de->d_name);

		if (de->d_name[0] == '.' || strchr(de->d_name, '.'))
			continue;
		if (!strncmp(de->d_name, "domain", 6))
			continue;
		if (len > 2 && !strcmp(de->d_name + len - 2, "-0"))
			continue;
		if (strchr(de->d_name, '-'))
			n++;
	}
	closedir(d);
	return n;
}

int odl_cli_run_diag(const struct odl_cli_params *params)
{
	char buf[256];
	struct lane_summary lanes;
	bool is_root = geteuid() == 0;
	bool have_tb_dbg, loopback = false;
	int xdomains, devices = 0, i;
	char odl_state[32] = "";
	int login_retries = -1;

	printf("OdinLink link diagnosis\n");
	printf("=======================\n\n");

	/* 1. Thunderbolt controller + driver */
	if (!file_exists(TB_SYS)) {
		printf("[!!] No Thunderbolt/USB4 bus.\n\n");
		printf("Diagnosis: the thunderbolt driver is not loaded or this\n"
		       "machine has no USB4 controller. Try: sudo modprobe thunderbolt\n");
		return 1;
	}
	printf("[ok] Thunderbolt/USB4 subsystem present\n");

	/* 2. OdinLink module */
	if (!file_exists("/sys/module/odl_tb5")) {
		printf("[!!] odl_tb5 kernel module not loaded.\n\n");
		printf("Diagnosis: load the driver first: sudo modprobe odl_tb5\n");
		return 1;
	}
	if (read_line("/sys/module/odl_tb5/parameters/loopback", buf,
		      sizeof(buf)) == 0 && atoi(buf) > 0)
		loopback = true;
	printf("[ok] odl_tb5 module loaded%s\n",
	       loopback ? " (loopback mode)" : "");

	/* 3. Physical link (skipped in loopback mode) */
	have_tb_dbg = file_exists(TB_DBG);
	if (!loopback) {
		if (have_tb_dbg && is_root) {
			scan_lanes(&lanes, params->verbose);
			switch (state_rank(lanes.best_state)) {
			case 3:
				printf("[ok] Link trained: %s port%d at %s\n",
				       lanes.router, lanes.port,
				       lane_speed_name((lanes.reg >> 16) & 0xf));
				break;
			case 2:
				printf("[!!] Cable detected on %s port%d but the "
				       "link is stuck in training%s\n",
				       lanes.router, lanes.port,
				       lanes.flapping ?
				       " (and cycling: train/drop/train)" : "");
				printf("\nDiagnosis: both ends see the cable but "
				       "link training never completes.\n"
				       "This is almost always the physical path: "
				       "reseat the cable firmly at both\n"
				       "ends, try the other port, or replace the "
				       "cable. If this started right\n"
				       "after a driver reload or one machine "
				       "rebooted, set thunderbolt\n"
				       "host_reset=false (a host router reset on "
				       "one side wedges the peer).\n");
				return 1;
			case 1:
				printf("[!!] No cable detected on any USB4 port\n");
				printf("\nDiagnosis: no peer is seen at the "
				       "connector level. Plug/replug the\n"
				       "cable. If it is already plugged in, the "
				       "connector logic may have been\n"
				       "wedged by a host router reset — replug "
				       "once and boot with\n"
				       "thunderbolt host_reset=false to keep it "
				       "from recurring.\n");
				return 1;
			default:
				printf("[??] Could not read lane state\n");
				break;
			}
		} else {
			int up = scan_lanes_sysfs();

			if (up)
				printf("[ok] %d USB4 link(s) up\n", up);
			else
				printf("[!!] No USB4 link up (run as root for "
				       "detailed lane state)\n");
		}

		/* 4. Peer discovery */
		xdomains = count_xdomains();
		if (xdomains) {
			printf("[ok] %d Thunderbolt peer(s) discovered (XDomain)\n",
			       xdomains);
		} else {
			printf("[!!] No peer discovered on the Thunderbolt bus\n");
			printf("\nDiagnosis: the link may be up but the peer has "
			       "not appeared. Check the\n"
			       "OTHER machine: is it booted, is thunderbolt "
			       "loaded there, does its own\n"
			       "'odl_tb5_cli diag' see this machine?\n");
			return 1;
		}
	}

	/* 5. Driver internal state (root + debugfs only) */
	if (is_root && file_exists(ODL_DBG)) {
		FILE *f = fopen(ODL_DBG, "r");

		if (f) {
			while (fgets(buf, sizeof(buf), f)) {
				if (params->verbose)
					printf("    %s", buf);
				sscanf(buf, "devices: %d", &devices);
				if (!strncmp(buf, "dev", 3)) {
					char *p = strstr(buf, "state=");

					if (p)
						sscanf(p, "state=%31s",
						       odl_state);
					p = strstr(buf, "login_retries=");
					if (p)
						sscanf(p, "login_retries=%d",
						       &login_retries);
				}
			}
			fclose(f);
		}
	}

	/* 6. Device nodes + userspace view */
	for (i = 0; i < 8; i++) {
		struct odl_tb5_peer_info info;
		char dev_path[32];
		int fd;

		snprintf(dev_path, sizeof(dev_path), "/dev/odl_tb5_%d", i);
		fd = open(dev_path, O_RDWR);
		if (fd < 0)
			continue;
		devices++;
		if (ioctl(fd, ODL_TB5_IOCTL_GET_PEER, &info) == 0) {
			static const char *names[] = {
				"DISCONNECTED", "HANDSHAKE", "CONNECTED",
				"ERROR", "READY"
			};
			const char *st = info.state < 5 ?
					 names[info.state] : "?";

			printf("[%s] %s: state=%s speed=%u Gb/s\n",
			       info.state == ODL_TB5_STATE_READY ? "ok" : "!!",
			       dev_path, st, info.link_speed);
			if (info.state == ODL_TB5_STATE_READY) {
				close(fd);
				printf("\nDiagnosis: all layers healthy — "
				       "OdinLink is READY on %s.\n", dev_path);
				return 0;
			}
		}
		close(fd);
	}

	if (!devices) {
		printf("[!!] Peer discovered but no /dev/odl_tb5_* device\n");
		printf("\nDiagnosis: the Thunderbolt peer is visible but the "
		       "OdinLink service did\n"
		       "not bind. Both machines must run odl_tb5 with the same "
		       "protocol setting\n"
		       "and compatible driver revisions. Check dmesg for "
		       "odl_tb5 messages.\n");
		return 1;
	}

	printf("\nDiagnosis: device exists but is not READY (state=%s",
	       odl_state[0] ? odl_state : "unknown");
	if (login_retries >= 0)
		printf(", login retries=%d", login_retries);
	printf(").\nThe login handshake is still running or gave up. Check "
	       "dmesg on BOTH\n"
	       "machines for 'protocol version' mismatches, and reload the "
	       "module to\n"
	       "restart the handshake: sudo modprobe -r odl_tb5 && sudo "
	       "modprobe odl_tb5\n");
	return 1;
}
