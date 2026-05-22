#ifndef NCCL_NET_V5_H_
#define NCCL_NET_V5_H_

#include <stdint.h>
#include <stddef.h>

#define NCCL_NET_MAX_STR_LEN 128

#define NCCL_PTR_HOST 0
#define NCCL_PTR_CUDA 2

typedef enum {
  ncclSuccess = 0,
  ncclUnhandledCudaError = 1,
  ncclSystemError = 2,
  ncclInternalError = 3,
  ncclInvalidArgument = 4,
  ncclInvalidUsage = 5,
} ncclResult_t;

typedef void (*ncclDebugLogger_t)(int level, const char* fmt, ...);

typedef struct ncclNetProperties_v4 {
  char name[NCCL_NET_MAX_STR_LEN];
  char pciPath[NCCL_NET_MAX_STR_LEN];
  uint64_t guid;
  int ptrSupport;
  int speed;
  int port;
  int maxComm;
} ncclNetProperties_v4_t;

typedef struct ncclNetProperties_v5 {
  char name[NCCL_NET_MAX_STR_LEN];
  char pciPath[NCCL_NET_MAX_STR_LEN];
  uint64_t guid;
  int ptrSupport;
  int speed;
  int port;
  int maxComm;
  int maxRecvs;
} ncclNetProperties_v5_t;

typedef struct ncclNet_v4 {
  const char* name;
  ncclResult_t (*init)(ncclDebugLogger_t logFunction);
  ncclResult_t (*devices)(int* ndev);
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_v4_t* props);
  ncclResult_t (*listen)(int dev, void* handle, void** listenComm);
  ncclResult_t (*connect)(int dev, void* handle, void** sendComm, void** recvComm);
  ncclResult_t (*accept)(void* listenComm, void** recvComm, void** sendComm);
  ncclResult_t (*regMr)(void* comm, void* data, int size, int type, void** mhandle);
  ncclResult_t (*deregMr)(void* comm, void* mhandle);
  ncclResult_t (*isend)(void* sendComm, void* data, int size, int tag, void** request);
  ncclResult_t (*irecv)(void* recvComm, int n, void** data, int* sizes, int* tags, void*** request);
  ncclResult_t (*iflush)(void* recvComm, int n, void** data, int* sizes, void*** request);
  ncclResult_t (*test)(void* request, int* done, int* size);
  ncclResult_t (*closeSend)(void* sendComm);
  ncclResult_t (*closeRecv)(void* recvComm);
  ncclResult_t (*closeListen)(void* listenComm);
} ncclNet_v4_t;

typedef struct ncclNet_v5 {
  const char* name;
  ncclResult_t (*init)(ncclDebugLogger_t logFunction);
  ncclResult_t (*devices)(int* ndev);
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_v5_t* props);
  ncclResult_t (*listen)(int dev, void* handle, void** listenComm);
  ncclResult_t (*connect)(int dev, void* handle, void** sendComm, void** recvComm);
  ncclResult_t (*accept)(void* listenComm, void** recvComm, void** sendComm);
  ncclResult_t (*regMr)(void* comm, void* data, size_t size, int type, void** mhandle);
  ncclResult_t (*deregMr)(void* comm, void* mhandle);
  ncclResult_t (*isend)(void* sendComm, void* data, int size, int tag, void* mhandle, void** request);
  ncclResult_t (*irecv)(void* recvComm, void* data, int size, int tag, void* mhandle, void** request);
  ncclResult_t (*iflush)(void* recvComm, void* data, int size, void* mhandle, void** request);
  ncclResult_t (*test)(void* request, int* done, int* size);
  ncclResult_t (*closeSend)(void* sendComm);
  ncclResult_t (*closeRecv)(void* recvComm);
  ncclResult_t (*closeListen)(void* listenComm);
  ncclResult_t (*getMr)(void* comm, void* data, size_t size, void* mhandle, void** mr);
} ncclNet_v5_t;

#endif
