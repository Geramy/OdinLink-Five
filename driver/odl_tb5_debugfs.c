// SPDX-License-Identifier: GPL-2.0
/*
 * OdinLink — Driver Self-Diagnosis (/sys/kernel/debug/odl_tb5/status)
 *
 * Dumps the one thing only the kernel side knows: how far each device got
 * through the connection handshake, and why it stopped. Userspace tooling
 * (odl_tb5_cli diag) combines this with the Thunderbolt bus state to give
 * a full picture of where link bring-up is stuck.
 */
#include <linux/debugfs.h>
#include <linux/seq_file.h>

#include "odl_tb5_core.h"

static struct dentry *odl_tb5_debugfs_root;

static const char *odl_tb5_state_name(enum odl_tb5_conn_state state)
{
	switch (state) {
	case ODL_TB5_STATE_DISCONNECTED: return "DISCONNECTED";
	case ODL_TB5_STATE_HANDSHAKE:    return "HANDSHAKE";
	case ODL_TB5_STATE_CONNECTED:    return "CONNECTED";
	case ODL_TB5_STATE_ERROR:        return "ERROR";
	case ODL_TB5_STATE_READY:        return "READY";
	}
	return "UNKNOWN";
}

static int odl_tb5_status_show(struct seq_file *s, void *unused)
{
	struct odl_tb5_device *dev;
	int count = 0;

	seq_printf(s, "loopback: %d\n", odl_loopback_count);

	mutex_lock(&odl_tb5_devices_lock);
	list_for_each_entry(dev, &odl_tb5_devices_list, list) {
		seq_printf(s, "dev%d: state=%s", dev->index,
			   odl_tb5_state_name(dev->state));
		seq_printf(s, " transport=%s",
			   dev->transport ? dev->transport->name : "none");
#if IS_ENABLED(CONFIG_USB4)
		if (dev->xd)
			seq_printf(s, " xdomain_route=0x%llx",
				   (unsigned long long)dev->xd->route);
#endif
		seq_printf(s, " login_sent=%d login_received=%d login_retries=%d",
			   dev->login_sent, dev->login_received,
			   dev->login_retries);
		seq_printf(s, " dma_verified=%d peer_ping_answered=%d",
			   dev->pong_received, dev->peer_ping_answered);
		seq_printf(s, " open_count=%d\n",
			   atomic_read(&dev->open_count));
		count++;
	}
	mutex_unlock(&odl_tb5_devices_lock);

	seq_printf(s, "devices: %d\n", count);
	return 0;
}
DEFINE_SHOW_ATTRIBUTE(odl_tb5_status);

void odl_tb5_debugfs_init(void)
{
	odl_tb5_debugfs_root = debugfs_create_dir("odl_tb5", NULL);
	debugfs_create_file("status", 0444, odl_tb5_debugfs_root, NULL,
			    &odl_tb5_status_fops);
}

void odl_tb5_debugfs_exit(void)
{
	debugfs_remove_recursive(odl_tb5_debugfs_root);
	odl_tb5_debugfs_root = NULL;
}
