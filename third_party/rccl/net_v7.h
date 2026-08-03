#ifndef RCCL_NET_V7_H_
#define RCCL_NET_V7_H_

#include <stdint.h>
#include <stddef.h>

#define MAX_STR_LEN 128

typedef enum {
  rcclSuccess = 0,
  rcclUnhandledCudaError = 1,
  rcclSystemError = 2,
  rcclInternalError = 3,
  rcclInvalidArgument = 4,
  rcclInvalidUsage = 5,
  rcclNumResults = 6
} rcclResult_t;

typedef void (*rcclDebugLogger_t)(int level, const char* fmt, ...);

/* Memory pointer types (bitmask in properties.ptrSupport). */
#define NCCL_PTR_HOST 0x1
#define NCCL_PTR_CUDA 0x2
#define NCCL_PTR_DMABUF 0x4

typedef struct rcclNetProperties_v7 {
  char* name;        /* NCCL v7: pointer, NOT an inline buffer */
  char* pciPath;
  uint64_t guid;
  int ptrSupport;
  int speed;
  int port;
  float latency;
  int maxComm;
  int maxRecvs;
  int netDeviceType;
  int netDeviceVersion;
} rcclNetProperties_v7_t;

/* Opaque device-side comm handle (unused by this CPU-staged plugin). */
typedef struct rcclNetDeviceHandle_v7 rcclNetDeviceHandle_v7_t;

/*
 * NCCL/RCCL net plugin API, version 7.  The field order and function
 * signatures MUST match RCCL's ncclNet_v7_t exactly — RCCL calls each
 * slot by offset with these argument lists.  In particular regMr /
 * regMrDmaBuf / deregMr sit between accept and isend, isend/irecv carry
 * an mhandle, irecv/iflush are multi-buffer (n, arrays), and the table
 * ends with getDeviceMr / irecvConsumed.
 */
typedef struct rcclNet_v7 {
  const char* name;
  rcclResult_t (*init)(rcclDebugLogger_t logFunction);
  rcclResult_t (*devices)(int* ndev);
  rcclResult_t (*getProperties)(int dev, rcclNetProperties_v7_t* props);
  rcclResult_t (*listen)(int dev, void* handle, void** listenComm);
  rcclResult_t (*connect)(int dev, void* handle, void** sendComm,
                          rcclNetDeviceHandle_v7_t** sendDevComm);
  rcclResult_t (*accept)(void* listenComm, void** recvComm,
                         rcclNetDeviceHandle_v7_t** recvDevComm);
  rcclResult_t (*regMr)(void* comm, void* data, int size, int type,
                        void** mhandle);
  rcclResult_t (*regMrDmaBuf)(void* comm, void* data, size_t size, int type,
                              uint64_t offset, int fd, void** mhandle);
  rcclResult_t (*deregMr)(void* comm, void* mhandle);
  rcclResult_t (*isend)(void* sendComm, void* data, int size, int tag,
                        void* mhandle, void** request);
  rcclResult_t (*irecv)(void* recvComm, int n, void** data, int* sizes,
                        int* tags, void** mhandles, void** request);
  rcclResult_t (*iflush)(void* recvComm, int n, void** data, int* sizes,
                         void** mhandles, void** request);
  rcclResult_t (*test)(void* request, int* done, int* sizes);
  rcclResult_t (*closeSend)(void* sendComm);
  rcclResult_t (*closeRecv)(void* recvComm);
  rcclResult_t (*closeListen)(void* listenComm);
  rcclResult_t (*getDeviceMr)(void* comm, void* mhandle, void** dptr_mhandle);
  rcclResult_t (*irecvConsumed)(void* recvComm, int n, void* request);
} rcclNet_v7_t;

#endif // RCCL_NET_V7_H_
