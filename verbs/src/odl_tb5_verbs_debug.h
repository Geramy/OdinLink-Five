#ifndef ODL_TB5_VERBS_DEBUG_H
#define ODL_TB5_VERBS_DEBUG_H

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <errno.h>
#include <string.h>
#include <time.h>

/*
 * OdinLink — Verbs: Tracing Every Call (for debugging)
 *
 * Set ODL_VERBS_DEBUG to see every ibv_* call logged to stderr:
 *   export ODL_VERBS_DEBUG=1
 * Levels: 0=off, 1=errors only, 5=everything (entry/exit + params)
 */

extern int odl_verbs_debug_level;

static inline int odl_verbs_get_debug(void)
{
    static int cached = -1;
    if (cached < 0) {
        char *env = getenv("ODL_VERBS_DEBUG");
        cached = env ? atoi(env) : 0;
        if (cached > 5) cached = 5;
    }
    return cached;
}

#define odl_log(level, fmt, ...) do { \
    if (odl_verbs_get_debug() >= (level)) { \
        struct timespec _ts; \
        clock_gettime(CLOCK_MONOTONIC, &_ts); \
        fprintf(stderr, "[ODL-VERBS %ld.%06ld] %s:%d: " fmt "\n", \
                _ts.tv_sec, _ts.tv_nsec / 1000, \
                __func__, __LINE__, ##__VA_ARGS__); \
    } \
} while (0)

#define odl_logerr(fmt, ...) odl_log(1, "ERROR: " fmt, ##__VA_ARGS__)
#define odl_logwarn(fmt, ...) odl_log(2, "WARN: " fmt, ##__VA_ARGS__)
#define odl_loginfo(fmt, ...) odl_log(3, fmt, ##__VA_ARGS__)
#define odl_logverbose(fmt, ...) odl_log(4, fmt, ##__VA_ARGS__)

/* Trace entry/exit of a verbs function at level 5 */
#define ODL_TRACE_ENTRY() \
    odl_log(5, "-> enter") \

#define ODL_TRACE_EXIT() \
    odl_log(5, "<- exit")

#define ODL_TRACE_EXIT_VAL(val) do { \
    odl_log(5, "<- exit (%d)", (int)(val)); \
    return (val); \
} while (0)

#define ODL_TRACE_EXIT_PTR(ptr) do { \
    odl_log(5, "<- exit (%p)", (void*)(ptr)); \
    return (ptr); \
} while (0)

/* Assertions that log and abort */
#define ODL_ASSERT(cond, fmt, ...) do { \
    if (!(cond)) { \
        odl_logerr("ASSERTION FAILED: " fmt, ##__VA_ARGS__); \
        fprintf(stderr, "[ODL-VERBS] ASSERT %s:%d: %s\n", \
                __func__, __LINE__, #cond); \
        abort(); \
    } \
} while (0)

/* Parameter validation that returns -EINVAL */
#define ODL_RETURN_EINVAL_IF(cond, fmt, ...) do { \
    if (cond) { \
        odl_logerr("EINVAL (" fmt ")", ##__VA_ARGS__); \
        return -EINVAL; \
    } \
} while (0)

#define ODL_RETURN_NULL_IF(cond, fmt, ...) do { \
    if (cond) { \
        odl_logerr("EINVAL (" fmt ")", ##__VA_ARGS__); \
        return NULL; \
    } \
} while (0)

#endif /* ODL_TB5_VERBS_DEBUG_H */
