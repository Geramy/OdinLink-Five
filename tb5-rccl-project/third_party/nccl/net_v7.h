#ifndef NCCL_NET_V7_H_
#define NCCL_NET_V7_H_

#include <stdint.h>
#include <stddef.h>

#define MAX_STR_LEN 128

typedef enum {
  ncclSuccess = 0,
  ncclUnhandledCudaError = 1,
  ncclSystemError = 2,
  ncclInternalError = 3,
  ncclInvalidArgument = 4,
  ncclInvalidUsage = 5,
  ncclNumResults = 6
} ncclResult_t;

typedef void (*ncclDebugLogger_t)(int level, const char* fmt, ...);

typedef struct ncclNetProperties_v7 ncclNetProperties_v7_t;

typedef struct ncclNet_v7 {
  // Name of the network (mainly for logs)
  const char* name;
  // Initialize the network.
  ncclResult_t (*init)(ncclDebugLogger_t logFunction);
  // Return the number of adapters.
  ncclResult_t (*devices)(int* ndev);
  // Get various device properties.
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_v7_t* props);
  // Create a listening socket.
  ncclResult_t (*listen)(int dev, void* handle, void** listenComm);
  // Connect to a handle.
  ncclResult_t (*connect)(int dev, void* handle, void** sendComm, void** recvComm);
  // Accept an incoming connection.
  ncclResult_t (*accept)(void* listenComm, void** recvComm, void** sendComm);
  // Close a listening socket.
  ncclResult_t (*closeListen)(void* listenComm);
  // Send data to the peer.
  ncclResult_t (*isend)(void* sendComm, void* data, int size, int tag, void** request);
  // Receive data from the peer.
  ncclResult_t (*irecv)(void* recvComm, void* data, int size, int tag, void** request);
  // Flush outstanding sends.
  ncclResult_t (*iflush)(void* sendComm, int dev, void** request);
  // Test whether a request is complete.
  ncclResult_t (*test)(void* request, int* done, int* size);
  // Close a send communicator.
  ncclResult_t (*closeSend)(void* sendComm);
  // Close a recv communicator.
  ncclResult_t (*closeRecv)(void* recvComm);
} ncclNet_v7_t;

typedef struct ncclNetProperties_v7 {
  char name[MAX_STR_LEN];
  char pciPath[MAX_STR_LEN];
  uint64_t guid;
  int ptrSupport;
  int speed;
  int port;
  int maxComm;
  int maxRecvs;
} ncclNetProperties_v7_t;

#endif // NCCL_NET_V7_H_
