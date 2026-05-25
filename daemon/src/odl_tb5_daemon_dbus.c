/*
 * OdinLink — Daemon: D-Bus API (Other Apps Ask Us Questions)
 *
 * Exposes peer status, connection info, and link statistics over
 * D-Bus so system tools, desktop widgets, and monitoring scripts
 * can ask "is the Thunderbolt link up?" without talking to the
 * kernel directly.
 */
#include "odl_tb5_daemon_dbus.h"
#include "odl_tb5_daemon_monitor.h"
#include "odl_tb5_daemon_config.h"
#include "odl_tb5_daemon_rccl_stats.h"
#include "odl_tb5_daemon_test.h"
#include "odl_tb5_daemon_sync.h"
#include "odl_tb5_daemon_sysinfo.h"
#include <stdio.h>
#include <string.h>

#define ODL_DBUS_NAME       "com.odinlink.Tb5Daemon"
#define ODL_DBUS_PATH       "/com/odinlink/Tb5Daemon"
#define ODL_DBUS_IFACE      "com.odinlink.Tb5Daemon"
#define ODL_DAEMON_VERSION  "1.0.0"

static GDBusConnection *dbus_conn;
static guint            dbus_owner_id;
static guint            dbus_reg_id;
static GDBusNodeInfo   *introspection_data;
static GMainLoop       *daemon_loop;

static const gchar introspection_xml[] =
	"<!DOCTYPE node PUBLIC \"-//freedesktop//DTD D-BUS Object Introspection 1.0//EN\"\n"
	" \"http://www.freedesktop.org/standards/dbus/1.0/introspect.dtd\">\n"
	"<node>\n"
	"  <interface name=\"com.odinlink.Tb5Daemon\">\n"
	"\n"
	"    <method name=\"GetDevices\">\n"
	"      <arg name=\"devices\" type=\"a(iss)\" direction=\"out\"/>\n"
	"    </method>\n"
	"\n"
	"    <method name=\"GetPeerInfo\">\n"
	"      <arg name=\"device_index\" type=\"i\" direction=\"in\"/>\n"
	"      <arg name=\"uuid\" type=\"s\" direction=\"out\"/>\n"
	"      <arg name=\"link_speed_gbps\" type=\"u\" direction=\"out\"/>\n"
	"      <arg name=\"link_width\" type=\"u\" direction=\"out\"/>\n"
	"      <arg name=\"state\" type=\"s\" direction=\"out\"/>\n"
	"      <arg name=\"vendor_name\" type=\"s\" direction=\"out\"/>\n"
	"      <arg name=\"device_name\" type=\"s\" direction=\"out\"/>\n"
	"    </method>\n"
	"\n"
	"    <method name=\"RunTest\">\n"
	"      <arg name=\"device_index\" type=\"i\" direction=\"in\"/>\n"
	"      <arg name=\"test_type\" type=\"s\" direction=\"in\"/>\n"
	"      <arg name=\"test_id\" type=\"s\" direction=\"out\"/>\n"
	"    </method>\n"
	"\n"
	"    <method name=\"CancelTest\">\n"
	"      <arg name=\"test_id\" type=\"s\" direction=\"in\"/>\n"
	"      <arg name=\"success\" type=\"b\" direction=\"out\"/>\n"
	"    </method>\n"
	"\n"
	"    <method name=\"GetTestStatus\">\n"
	"      <arg name=\"test_id\" type=\"s\" direction=\"in\"/>\n"
	"      <arg name=\"state\" type=\"s\" direction=\"out\"/>\n"
	"      <arg name=\"progress_pct\" type=\"u\" direction=\"out\"/>\n"
	"      <arg name=\"current_subtest\" type=\"s\" direction=\"out\"/>\n"
	"    </method>\n"
	"\n"
	"    <method name=\"GetTestResult\">\n"
	"      <arg name=\"test_id\" type=\"s\" direction=\"in\"/>\n"
	"      <arg name=\"result_json\" type=\"s\" direction=\"out\"/>\n"
	"    </method>\n"
	"\n"
	"    <method name=\"SetSyncFolder\">\n"
	"      <arg name=\"path\" type=\"s\" direction=\"in\"/>\n"
	"      <arg name=\"success\" type=\"b\" direction=\"out\"/>\n"
	"    </method>\n"
	"\n"
	"    <method name=\"GetSyncFolder\">\n"
	"      <arg name=\"path\" type=\"s\" direction=\"out\"/>\n"
	"    </method>\n"
	"\n"
	"    <method name=\"GetSyncStatus\">\n"
	"      <arg name=\"enabled\" type=\"b\" direction=\"out\"/>\n"
	"      <arg name=\"files_pending\" type=\"u\" direction=\"out\"/>\n"
	"      <arg name=\"bytes_transferred\" type=\"t\" direction=\"out\"/>\n"
	"      <arg name=\"last_sync_time\" type=\"s\" direction=\"out\"/>\n"
	"    </method>\n"
	"\n"
	"    <method name=\"SetSyncEnabled\">\n"
	"      <arg name=\"enabled\" type=\"b\" direction=\"in\"/>\n"
	"    </method>\n"
	"\n"
	"    <method name=\"ListFiles\">\n"
	"      <arg name=\"dir_path\" type=\"s\" direction=\"in\"/>\n"
	"      <arg name=\"files_json\" type=\"s\" direction=\"out\"/>\n"
	"    </method>\n"
	"\n"
	"    <method name=\"FetchFile\">\n"
	"      <arg name=\"rel_path\" type=\"s\" direction=\"in\"/>\n"
	"      <arg name=\"success\" type=\"b\" direction=\"out\"/>\n"
	"    </method>\n"
	"\n"
	"    <method name=\"TransferFile\">\n"
	"      <arg name=\"rel_path\" type=\"s\" direction=\"in\"/>\n"
	"      <arg name=\"success\" type=\"b\" direction=\"out\"/>\n"
	"    </method>\n"
	"\n"
	"    <method name=\"RemoveLocalCopy\">\n"
	"      <arg name=\"rel_path\" type=\"s\" direction=\"in\"/>\n"
	"      <arg name=\"success\" type=\"b\" direction=\"out\"/>\n"
	"    </method>\n"
	"\n"
	"    <method name=\"RemoveFromPeer\">\n"
	"      <arg name=\"rel_path\" type=\"s\" direction=\"in\"/>\n"
	"      <arg name=\"success\" type=\"b\" direction=\"out\"/>\n"
	"    </method>\n"
	"\n"
	"    <method name=\"GetRcclStats\">\n"
	"      <arg name=\"stats_json\" type=\"s\" direction=\"out\"/>\n"
	"    </method>\n"
	"\n"
	"    <method name=\"GetVersion\">\n"
	"      <arg name=\"version\" type=\"s\" direction=\"out\"/>\n"
	"    </method>\n"
	"\n"
	"    <method name=\"GetLocalSystemInfo\">\n"
	"      <arg name=\"cpus\" type=\"a(suuu)\" direction=\"out\"/>\n"
	"      <arg name=\"ram_total_mb\" type=\"u\" direction=\"out\"/>\n"
	"      <arg name=\"ram_available_mb\" type=\"u\" direction=\"out\"/>\n"
	"      <arg name=\"gpus\" type=\"a(suu)\" direction=\"out\"/>\n"
	"    </method>\n"
	"\n"
	"    <method name=\"GetPeerSystemInfo\">\n"
	"      <arg name=\"device_index\" type=\"i\" direction=\"in\"/>\n"
	"      <arg name=\"cpus\" type=\"a(suuu)\" direction=\"out\"/>\n"
	"      <arg name=\"ram_total_mb\" type=\"u\" direction=\"out\"/>\n"
	"      <arg name=\"ram_available_mb\" type=\"u\" direction=\"out\"/>\n"
	"      <arg name=\"gpus\" type=\"a(suu)\" direction=\"out\"/>\n"
	"    </method>\n"
	"\n"
	"    <method name=\"GetRecentLogs\">\n"
	"      <arg name=\"max_lines\" type=\"u\" direction=\"in\"/>\n"
	"      <arg name=\"log_text\" type=\"s\" direction=\"out\"/>\n"
	"    </method>\n"
	"\n"
	"    <signal name=\"DeviceAdded\">\n"
	"      <arg name=\"device_index\" type=\"i\"/>\n"
	"      <arg name=\"device_name\" type=\"s\"/>\n"
	"    </signal>\n"
	"\n"
	"    <signal name=\"DeviceRemoved\">\n"
	"      <arg name=\"device_index\" type=\"i\"/>\n"
	"    </signal>\n"
	"\n"
	"    <signal name=\"PeerStateChanged\">\n"
	"      <arg name=\"device_index\" type=\"i\"/>\n"
	"      <arg name=\"new_state\" type=\"s\"/>\n"
	"    </signal>\n"
	"\n"
	"    <signal name=\"TestProgress\">\n"
	"      <arg name=\"test_id\" type=\"s\"/>\n"
	"      <arg name=\"progress_pct\" type=\"u\"/>\n"
	"      <arg name=\"current_subtest\" type=\"s\"/>\n"
	"    </signal>\n"
	"\n"
	"    <signal name=\"TestCompleted\">\n"
	"      <arg name=\"test_id\" type=\"s\"/>\n"
	"      <arg name=\"success\" type=\"b\"/>\n"
	"      <arg name=\"summary\" type=\"s\"/>\n"
	"    </signal>\n"
	"\n"
	"    <signal name=\"SyncFileTransferred\">\n"
	"      <arg name=\"filename\" type=\"s\"/>\n"
	"      <arg name=\"direction\" type=\"s\"/>\n"
	"      <arg name=\"bytes\" type=\"t\"/>\n"
	"    </signal>\n"
	"\n"
	"    <signal name=\"SyncConflict\">\n"
	"      <arg name=\"filename\" type=\"s\"/>\n"
	"      <arg name=\"resolution\" type=\"s\"/>\n"
	"    </signal>\n"
	"\n"
	"    <property name=\"DaemonState\" type=\"s\" access=\"read\"/>\n"
	"    <property name=\"ConnectedPeerCount\" type=\"u\" access=\"read\"/>\n"
	"    <property name=\"SyncEnabled\" type=\"b\" access=\"readwrite\"/>\n"
	"\n"
	"  </interface>\n"
	"</node>\n";

static void format_uuid_str(const uint8_t uuid[16], char *buf, size_t len)
{
	snprintf(buf, len,
		"%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
		uuid[0], uuid[1], uuid[2], uuid[3],
		uuid[4], uuid[5], uuid[6], uuid[7],
		uuid[8], uuid[9], uuid[10], uuid[11],
		uuid[12], uuid[13], uuid[14], uuid[15]);
}

static const char *test_state_str(int state)
{
	switch (state) {
	case ODL_DTEST_QUEUED:    return "queued";
	case ODL_DTEST_RUNNING:   return "running";
	case ODL_DTEST_COMPLETED: return "completed";
	case ODL_DTEST_FAILED:    return "failed";
	case ODL_DTEST_CANCELLED: return "cancelled";
	default:                  return "unknown";
	}
}

static void handle_get_devices(GDBusMethodInvocation *invocation)
{
	GVariantBuilder builder;
	g_variant_builder_init(&builder, G_VARIANT_TYPE("a(iss)"));

	g_mutex_lock(&g_device_table.lock);
	for (int i = 0; i < ODL_DAEMON_MAX_DEVICES; i++) {
		if (!g_device_table.slots[i].present)
			continue;

		const char *state = g_device_table.slots[i].state_str;
		const char *name = g_device_table.slots[i].peer.device_name;
		if (!name[0])
			name = "Unknown";

		g_variant_builder_add(&builder, "(iss)", i, state, name);
	}
	g_mutex_unlock(&g_device_table.lock);

	g_dbus_method_invocation_return_value(
		invocation,
		g_variant_new("(a(iss))", &builder));
}

static void handle_get_peer_info(GDBusMethodInvocation *invocation,
                                 GVariant *parameters)
{
	gint32 device_index;
	g_variant_get(parameters, "(i)", &device_index);

	if (device_index < 0 || device_index >= ODL_DAEMON_MAX_DEVICES) {
		g_dbus_method_invocation_return_dbus_error(
			invocation, "com.odinlink.Error.InvalidIndex",
			"Device index out of range");
		return;
	}

	g_mutex_lock(&g_device_table.lock);

	if (!g_device_table.slots[device_index].present) {
		g_mutex_unlock(&g_device_table.lock);
		g_dbus_method_invocation_return_dbus_error(
			invocation, "com.odinlink.Error.NotFound",
			"Device not present");
		return;
	}

	struct odl_daemon_device_slot *slot = &g_device_table.slots[device_index];
	struct odl_tb5_peer_info *peer = &slot->peer;

	char uuid_str[48];
	format_uuid_str(peer->uuid, uuid_str, sizeof(uuid_str));

	const char *state_str = odl_daemon_state_str(peer->state);
	const char *vendor = peer->vendor_name[0] ? peer->vendor_name : "";
	const char *devname = peer->device_name[0] ? peer->device_name : "";

	g_dbus_method_invocation_return_value(
		invocation,
		g_variant_new("(suusss)",
			uuid_str,
			(guint32)peer->link_speed,
			(guint32)peer->link_width,
			state_str,
			vendor,
			devname));

	g_mutex_unlock(&g_device_table.lock);
}

static void handle_run_test(GDBusMethodInvocation *invocation,
                            GVariant *parameters)
{
	gint32 device_index;
	const gchar *test_type;
	g_variant_get(parameters, "(i&s)", &device_index, &test_type);

	const char *test_id = odl_daemon_test_run(device_index, test_type);
	if (!test_id) {
		g_dbus_method_invocation_return_dbus_error(
			invocation, "com.odinlink.Error.NotImplemented",
			"Test execution not yet implemented (Phase 2)");
		return;
	}

	g_dbus_method_invocation_return_value(
		invocation,
		g_variant_new("(s)", test_id));
}

static void handle_cancel_test(GDBusMethodInvocation *invocation,
                               GVariant *parameters)
{
	const gchar *test_id;
	g_variant_get(parameters, "(&s)", &test_id);

	gboolean success = odl_daemon_test_cancel(test_id);

	g_dbus_method_invocation_return_value(
		invocation,
		g_variant_new("(b)", success));
}

static void handle_get_test_status(GDBusMethodInvocation *invocation,
                                   GVariant *parameters)
{
	const gchar *test_id;
	g_variant_get(parameters, "(&s)", &test_id);

	struct odl_daemon_test_ctx *ctx = odl_daemon_test_find(test_id);
	if (!ctx) {
		g_dbus_method_invocation_return_dbus_error(
			invocation, "com.odinlink.Error.NotFound",
			"Test ID not found");
		return;
	}

	g_dbus_method_invocation_return_value(
		invocation,
		g_variant_new("(sus)",
			test_state_str(ctx->state),
			(guint32)ctx->progress_pct,
			ctx->current_subtest[0] ? ctx->current_subtest : ""));
}

static void handle_get_test_result(GDBusMethodInvocation *invocation,
                                   GVariant *parameters)
{
	const gchar *test_id;
	g_variant_get(parameters, "(&s)", &test_id);

	struct odl_daemon_test_ctx *ctx = odl_daemon_test_find(test_id);
	if (!ctx) {
		g_dbus_method_invocation_return_dbus_error(
			invocation, "com.odinlink.Error.NotFound",
			"Test ID not found");
		return;
	}

	const char *result = ctx->result_json ? ctx->result_json : "{}";

	g_dbus_method_invocation_return_value(
		invocation,
		g_variant_new("(s)", result));
}

static void handle_set_sync_folder(GDBusMethodInvocation *invocation,
                                   GVariant *parameters)
{
	const gchar *path;
	g_variant_get(parameters, "(&s)", &path);

	int ret = odl_daemon_sync_set_folder(path);
	if (ret == 0) {
		snprintf(g_config.sync_folder, sizeof(g_config.sync_folder),
		         "%s", path);
		odl_daemon_config_save();
	}

	g_dbus_method_invocation_return_value(
		invocation,
		g_variant_new("(b)", (gboolean)(ret == 0)));
}

static void handle_get_sync_folder(GDBusMethodInvocation *invocation)
{
	g_dbus_method_invocation_return_value(
		invocation,
		g_variant_new("(s)", g_config.sync_folder));
}

static void handle_get_sync_status(GDBusMethodInvocation *invocation)
{
	g_mutex_lock(&g_sync_status.lock);

	g_dbus_method_invocation_return_value(
		invocation,
		g_variant_new("(buts)",
			(gboolean)g_sync_status.enabled,
			(guint32)g_sync_status.files_pending,
			(guint64)g_sync_status.bytes_transferred,
			g_sync_status.last_sync_time));

	g_mutex_unlock(&g_sync_status.lock);
}

static void handle_set_sync_enabled(GDBusMethodInvocation *invocation,
                                    GVariant *parameters)
{
	gboolean enabled;
	g_variant_get(parameters, "(b)", &enabled);

	odl_daemon_sync_set_enabled(enabled);
	g_config.sync_enabled = enabled;
	odl_daemon_config_save();

	g_dbus_method_invocation_return_value(invocation, NULL);
}

static void handle_list_files(GDBusMethodInvocation *invocation,
                              GVariant *parameters)
{
	const gchar *dir_path;
	g_variant_get(parameters, "(&s)", &dir_path);

	char *json = odl_daemon_sync_list_files_json(dir_path);
	if (!json)
		json = g_strdup("[]");

	g_dbus_method_invocation_return_value(
		invocation,
		g_variant_new("(s)", json));

	g_free(json);
}

static void handle_fetch_file(GDBusMethodInvocation *invocation,
                              GVariant *parameters)
{
	const gchar *rel_path;
	g_variant_get(parameters, "(&s)", &rel_path);

	int ret = odl_daemon_sync_fetch_file(rel_path);

	g_dbus_method_invocation_return_value(
		invocation,
		g_variant_new("(b)", (gboolean)(ret == 0)));
}

static void handle_transfer_file(GDBusMethodInvocation *invocation,
                                 GVariant *parameters)
{
	const gchar *rel_path;
	g_variant_get(parameters, "(&s)", &rel_path);

	int ret = odl_daemon_sync_transfer_file(rel_path);

	g_dbus_method_invocation_return_value(
		invocation,
		g_variant_new("(b)", (gboolean)(ret == 0)));
}

static void handle_remove_local_copy(GDBusMethodInvocation *invocation,
                                     GVariant *parameters)
{
	const gchar *rel_path;
	g_variant_get(parameters, "(&s)", &rel_path);

	int ret = odl_daemon_sync_remove_local(rel_path);

	g_dbus_method_invocation_return_value(
		invocation,
		g_variant_new("(b)", (gboolean)(ret == 0)));
}

static void handle_remove_from_peer(GDBusMethodInvocation *invocation,
                                    GVariant *parameters)
{
	const gchar *rel_path;
	g_variant_get(parameters, "(&s)", &rel_path);

	int ret = odl_daemon_sync_remove_from_peer(rel_path);

	g_dbus_method_invocation_return_value(
		invocation,
		g_variant_new("(b)", (gboolean)(ret == 0)));
}

static void handle_get_rccl_stats(GDBusMethodInvocation *invocation)
{
	char *json = odl_daemon_rccl_get_json();

	g_dbus_method_invocation_return_value(
		invocation,
		g_variant_new("(s)", json));

	g_free(json);
}

static void handle_get_version(GDBusMethodInvocation *invocation)
{
	g_dbus_method_invocation_return_value(
		invocation,
		g_variant_new("(s)", ODL_DAEMON_VERSION));
}

static void handle_get_local_system_info(GDBusMethodInvocation *invocation)
{
	struct odl_sysinfo si;
	odl_daemon_sysinfo_collect(&si);

	GVariantBuilder cpu_builder;
	g_variant_builder_init(&cpu_builder, G_VARIANT_TYPE("a(suuu)"));
	for (int i = 0; i < si.num_cpus; i++) {
		g_variant_builder_add(&cpu_builder, "(suuu)",
			si.cpus[i].model,
			(guint32)si.cpus[i].cores,
			(guint32)si.cpus[i].threads,
			(guint32)si.cpus[i].freq_mhz);
	}

	GVariantBuilder gpu_builder;
	g_variant_builder_init(&gpu_builder, G_VARIANT_TYPE("a(suu)"));
	for (int i = 0; i < si.num_gpus; i++) {
		g_variant_builder_add(&gpu_builder, "(suu)",
			si.gpus[i].name,
			(guint32)si.gpus[i].vram_total_mb,
			(guint32)si.gpus[i].vram_used_mb);
	}

	g_dbus_method_invocation_return_value(
		invocation,
		g_variant_new("(a(suuu)uua(suu))",
			&cpu_builder,
			(guint32)si.ram_total_mb,
			(guint32)si.ram_available_mb,
			&gpu_builder));
}

static void handle_get_peer_system_info(GDBusMethodInvocation *invocation,
					GVariant *parameters)
{
	gint32 device_index;
	g_variant_get(parameters, "(i)", &device_index);

	if (device_index < 0 || device_index >= ODL_DAEMON_MAX_DEVICES) {
		g_dbus_method_invocation_return_dbus_error(
			invocation, "com.odinlink.Error.InvalidIndex",
			"Device index out of range");
		return;
	}

	g_mutex_lock(&g_device_table.lock);
	struct odl_daemon_device_slot *slot =
		&g_device_table.slots[device_index];

	if (!slot->present) {
		g_mutex_unlock(&g_device_table.lock);
		g_dbus_method_invocation_return_dbus_error(
			invocation, "com.odinlink.Error.NotFound",
			"Device not present");
		return;
	}

	struct odl_sysinfo si;
	bool cached = slot->has_remote_sysinfo;
	if (cached)
		memcpy(&si, &slot->remote_sysinfo, sizeof(si));
	g_mutex_unlock(&g_device_table.lock);

	if (!cached) {
		int ret = odl_daemon_test_request_peer_sysinfo(
				device_index, &si);
		if (ret < 0) {
			g_dbus_method_invocation_return_dbus_error(
				invocation, "com.odinlink.Error.Unavailable",
				"Could not retrieve peer system info "
				"(peer may not be running OdinLink daemon)");
			return;
		}

		g_mutex_lock(&g_device_table.lock);
		memcpy(&g_device_table.slots[device_index].remote_sysinfo,
		       &si, sizeof(si));
		g_device_table.slots[device_index].has_remote_sysinfo = true;
		g_mutex_unlock(&g_device_table.lock);
	}

	GVariantBuilder cpu_builder;
	g_variant_builder_init(&cpu_builder, G_VARIANT_TYPE("a(suuu)"));
	for (int i = 0; i < si.num_cpus; i++) {
		g_variant_builder_add(&cpu_builder, "(suuu)",
			si.cpus[i].model,
			(guint32)si.cpus[i].cores,
			(guint32)si.cpus[i].threads,
			(guint32)si.cpus[i].freq_mhz);
	}

	GVariantBuilder gpu_builder;
	g_variant_builder_init(&gpu_builder, G_VARIANT_TYPE("a(suu)"));
	for (int i = 0; i < si.num_gpus; i++) {
		g_variant_builder_add(&gpu_builder, "(suu)",
			si.gpus[i].name,
			(guint32)si.gpus[i].vram_total_mb,
			(guint32)si.gpus[i].vram_used_mb);
	}

	g_dbus_method_invocation_return_value(
		invocation,
		g_variant_new("(a(suuu)uua(suu))",
			&cpu_builder,
			(guint32)si.ram_total_mb,
			(guint32)si.ram_available_mb,
			&gpu_builder));
}

static void handle_get_recent_logs(GDBusMethodInvocation *invocation,
                                   GVariant *parameters)
{
	guint32 max_lines;
	g_variant_get(parameters, "(u)", &max_lines);
	if (max_lines == 0 || max_lines > 500)
		max_lines = 100;

	char cmd[512];
	snprintf(cmd, sizeof(cmd),
		"{ dmesg --time-format iso 2>/dev/null | "
		"grep -i 'odinlink\\|odl_tb5' | tail -n %u; "
		"journalctl --user -u odl-tb5-daemon --no-pager "
		"-n %u --output=short-iso 2>/dev/null; } | "
		"sort | tail -n %u",
		max_lines, max_lines, max_lines);

	GString *result = g_string_new(NULL);
	FILE *fp = popen(cmd, "r");
	if (fp) {
		char line[512];
		while (fgets(line, sizeof(line), fp))
			g_string_append(result, line);
		pclose(fp);
	}

	if (result->len == 0)
		g_string_append(result, "(no recent OdinLink log entries)");

	g_dbus_method_invocation_return_value(
		invocation,
		g_variant_new("(s)", result->str));
	g_string_free(result, TRUE);
}

static void on_method_call(GDBusConnection       *connection,
                           const gchar           *sender,
                           const gchar           *object_path,
                           const gchar           *interface_name,
                           const gchar           *method_name,
                           GVariant              *parameters,
                           GDBusMethodInvocation *invocation,
                           gpointer               user_data)
{
	(void)connection;
	(void)sender;
	(void)object_path;
	(void)interface_name;
	(void)user_data;

	if (g_strcmp0(method_name, "GetDevices") == 0) {
		handle_get_devices(invocation);
	} else if (g_strcmp0(method_name, "GetPeerInfo") == 0) {
		handle_get_peer_info(invocation, parameters);
	} else if (g_strcmp0(method_name, "RunTest") == 0) {
		handle_run_test(invocation, parameters);
	} else if (g_strcmp0(method_name, "CancelTest") == 0) {
		handle_cancel_test(invocation, parameters);
	} else if (g_strcmp0(method_name, "GetTestStatus") == 0) {
		handle_get_test_status(invocation, parameters);
	} else if (g_strcmp0(method_name, "GetTestResult") == 0) {
		handle_get_test_result(invocation, parameters);
	} else if (g_strcmp0(method_name, "SetSyncFolder") == 0) {
		handle_set_sync_folder(invocation, parameters);
	} else if (g_strcmp0(method_name, "GetSyncFolder") == 0) {
		handle_get_sync_folder(invocation);
	} else if (g_strcmp0(method_name, "GetSyncStatus") == 0) {
		handle_get_sync_status(invocation);
	} else if (g_strcmp0(method_name, "SetSyncEnabled") == 0) {
		handle_set_sync_enabled(invocation, parameters);
	} else if (g_strcmp0(method_name, "ListFiles") == 0) {
		handle_list_files(invocation, parameters);
	} else if (g_strcmp0(method_name, "FetchFile") == 0) {
		handle_fetch_file(invocation, parameters);
	} else if (g_strcmp0(method_name, "TransferFile") == 0) {
		handle_transfer_file(invocation, parameters);
	} else if (g_strcmp0(method_name, "RemoveLocalCopy") == 0) {
		handle_remove_local_copy(invocation, parameters);
	} else if (g_strcmp0(method_name, "RemoveFromPeer") == 0) {
		handle_remove_from_peer(invocation, parameters);
	} else if (g_strcmp0(method_name, "GetRcclStats") == 0) {
		handle_get_rccl_stats(invocation);
	} else if (g_strcmp0(method_name, "GetVersion") == 0) {
		handle_get_version(invocation);
	} else if (g_strcmp0(method_name, "GetLocalSystemInfo") == 0) {
		handle_get_local_system_info(invocation);
	} else if (g_strcmp0(method_name, "GetPeerSystemInfo") == 0) {
		handle_get_peer_system_info(invocation, parameters);
	} else if (g_strcmp0(method_name, "GetRecentLogs") == 0) {
		handle_get_recent_logs(invocation, parameters);
	} else {
		g_dbus_method_invocation_return_dbus_error(
			invocation,
			"org.freedesktop.DBus.Error.UnknownMethod",
			"Unknown method");
	}
}

static GVariant *on_get_property(GDBusConnection  *connection,
                                 const gchar      *sender,
                                 const gchar      *object_path,
                                 const gchar      *interface_name,
                                 const gchar      *property_name,
                                 GError          **error,
                                 gpointer          user_data)
{
	(void)connection;
	(void)sender;
	(void)object_path;
	(void)interface_name;
	(void)user_data;

	if (g_strcmp0(property_name, "DaemonState") == 0) {
		return g_variant_new_string("running");
	} else if (g_strcmp0(property_name, "ConnectedPeerCount") == 0) {
		g_mutex_lock(&g_device_table.lock);
		guint32 count = (guint32)g_device_table.connected_count;
		g_mutex_unlock(&g_device_table.lock);
		return g_variant_new_uint32(count);
	} else if (g_strcmp0(property_name, "SyncEnabled") == 0) {
		g_mutex_lock(&g_sync_status.lock);
		gboolean enabled = g_sync_status.enabled;
		g_mutex_unlock(&g_sync_status.lock);
		return g_variant_new_boolean(enabled);
	}

	g_set_error(error, G_DBUS_ERROR, G_DBUS_ERROR_UNKNOWN_PROPERTY,
	            "Unknown property: %s", property_name);
	return NULL;
}

static gboolean on_set_property(GDBusConnection  *connection,
                                const gchar      *sender,
                                const gchar      *object_path,
                                const gchar      *interface_name,
                                const gchar      *property_name,
                                GVariant         *value,
                                GError          **error,
                                gpointer          user_data)
{
	(void)connection;
	(void)sender;
	(void)object_path;
	(void)interface_name;
	(void)user_data;

	if (g_strcmp0(property_name, "SyncEnabled") == 0) {
		gboolean enabled = g_variant_get_boolean(value);
		odl_daemon_sync_set_enabled(enabled);
		g_config.sync_enabled = enabled;
		odl_daemon_config_save();
		return TRUE;
	}

	g_set_error(error, G_DBUS_ERROR, G_DBUS_ERROR_PROPERTY_READ_ONLY,
	            "Property %s is read-only", property_name);
	return FALSE;
}

static const GDBusInterfaceVTable interface_vtable = {
	on_method_call,
	on_get_property,
	on_set_property,
};

static void on_bus_acquired(GDBusConnection *connection,
                            const gchar     *name,
                            gpointer         user_data)
{
	(void)name;
	(void)user_data;

	GError *err = NULL;

	dbus_conn = connection;

	dbus_reg_id = g_dbus_connection_register_object(
		connection,
		ODL_DBUS_PATH,
		introspection_data->interfaces[0],
		&interface_vtable,
		NULL,
		NULL,
		&err);

	if (dbus_reg_id == 0) {
		g_printerr("odl_tb5_daemon: failed to register D-Bus object: %s\n",
		           err->message);
		g_error_free(err);
	}
}

static void on_name_acquired(GDBusConnection *connection,
                              const gchar     *name,
                              gpointer         user_data)
{
	(void)connection;
	(void)user_data;
	g_printerr("odl_tb5_daemon: acquired D-Bus name %s\n", name);
}

static void on_name_lost(GDBusConnection *connection,
                          const gchar     *name,
                          gpointer         user_data)
{
	(void)connection;
	(void)user_data;
	g_printerr("odl_tb5_daemon: lost D-Bus name %s - shutting down\n", name);
	if (daemon_loop)
		g_main_loop_quit(daemon_loop);
}

int odl_daemon_dbus_init(GMainLoop *loop)
{
	daemon_loop = loop;

	introspection_data = g_dbus_node_info_new_for_xml(introspection_xml, NULL);
	if (!introspection_data) {
		g_printerr("odl_tb5_daemon: failed to parse introspection XML\n");
		return -1;
	}

	dbus_owner_id = g_bus_own_name(
		G_BUS_TYPE_SESSION,
		ODL_DBUS_NAME,
		G_BUS_NAME_OWNER_FLAGS_NONE,
		on_bus_acquired,
		on_name_acquired,
		on_name_lost,
		NULL,
		NULL);

	return 0;
}

void odl_daemon_dbus_shutdown(void)
{
	if (dbus_owner_id > 0) {
		g_bus_unown_name(dbus_owner_id);
		dbus_owner_id = 0;
	}

	if (dbus_reg_id > 0 && dbus_conn) {
		g_dbus_connection_unregister_object(dbus_conn, dbus_reg_id);
		dbus_reg_id = 0;
	}

	if (introspection_data) {
		g_dbus_node_info_unref(introspection_data);
		introspection_data = NULL;
	}

	dbus_conn = NULL;
}

GDBusConnection *odl_daemon_dbus_get_connection(void)
{
	return dbus_conn;
}

void odl_daemon_dbus_emit_device_added(int index, const char *name)
{
	if (!dbus_conn)
		return;

	g_dbus_connection_emit_signal(
		dbus_conn, NULL,
		ODL_DBUS_PATH, ODL_DBUS_IFACE,
		"DeviceAdded",
		g_variant_new("(is)", (gint32)index, name),
		NULL);
}

void odl_daemon_dbus_emit_device_removed(int index)
{
	if (!dbus_conn)
		return;

	g_dbus_connection_emit_signal(
		dbus_conn, NULL,
		ODL_DBUS_PATH, ODL_DBUS_IFACE,
		"DeviceRemoved",
		g_variant_new("(i)", (gint32)index),
		NULL);
}

void odl_daemon_dbus_emit_peer_state_changed(int index, const char *state)
{
	if (!dbus_conn)
		return;

	g_dbus_connection_emit_signal(
		dbus_conn, NULL,
		ODL_DBUS_PATH, ODL_DBUS_IFACE,
		"PeerStateChanged",
		g_variant_new("(is)", (gint32)index, state),
		NULL);
}

void odl_daemon_dbus_emit_test_progress(const char *test_id,
                                         unsigned progress,
                                         const char *subtest)
{
	if (!dbus_conn)
		return;

	g_dbus_connection_emit_signal(
		dbus_conn, NULL,
		ODL_DBUS_PATH, ODL_DBUS_IFACE,
		"TestProgress",
		g_variant_new("(sus)", test_id, (guint32)progress, subtest),
		NULL);
}

void odl_daemon_dbus_emit_test_completed(const char *test_id,
                                          gboolean success,
                                          const char *summary)
{
	if (!dbus_conn)
		return;

	g_dbus_connection_emit_signal(
		dbus_conn, NULL,
		ODL_DBUS_PATH, ODL_DBUS_IFACE,
		"TestCompleted",
		g_variant_new("(sbs)", test_id, success, summary),
		NULL);
}

void odl_daemon_dbus_emit_sync_file_transferred(const char *filename,
                                                  const char *direction,
                                                  uint64_t bytes)
{
	if (!dbus_conn)
		return;

	g_dbus_connection_emit_signal(
		dbus_conn, NULL,
		ODL_DBUS_PATH, ODL_DBUS_IFACE,
		"SyncFileTransferred",
		g_variant_new("(sst)", filename, direction, (guint64)bytes),
		NULL);
}

void odl_daemon_dbus_emit_sync_conflict(const char *filename,
                                         const char *resolution)
{
	if (!dbus_conn)
		return;

	g_dbus_connection_emit_signal(
		dbus_conn, NULL,
		ODL_DBUS_PATH, ODL_DBUS_IFACE,
		"SyncConflict",
		g_variant_new("(ss)", filename, resolution),
		NULL);
}
