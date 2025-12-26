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

typedef struct rcclNetProperties_v7 rcclNetProperties_v7_t;

typedef struct rcclNet_v7 {
  // Name of the network (mainly for logs)
  const char* name;
  // Initialize the network.
  rcclResult_t (*init)(rcclDebugLogger_t logFunction);
  // Return the number of adapters.
  rcclResult_t (*devices)(int* ndev);
  // Get various device properties.
  rcclResult_t (*getProperties)(int dev, rcclNetProperties_v7_t* props);
  // Create a listening socket.
  rcclResult_t (*listen)(int dev, void* handle, void** listenComm);
  // Connect to a handle.
  rcclResult_t (*connect)(int dev, void* handle, void** sendComm, void** recvComm);
  // Accept an incoming connection.
  rcclResult_t (*accept)(void* listenComm, void** recvComm, void** sendComm);
  // Close a listening socket.
  rcclResult_t (*closeListen)(void* listenComm);
  // Send data to the peer.
  rcclResult_t (*isend)(void* sendComm, void* data, int size, int tag, void** request);
  // Receive data from the peer.
  rcclResult_t (*irecv)(void* recvComm, void* data, int size, int tag, void** request);
  // Flush outstanding sends.
  rcclResult_t (*iflush)(void* sendComm, int dev, void** request);
  // Test whether a request is complete.
  rcclResult_t (*test)(void* request, int* done, int* size);
  // Close a send communicator.
  rcclResult_t (*closeSend)(void* sendComm);
  // Close a recv communicator.
  rcclResult_t (*closeRecv)(void* recvComm);
} rcclNet_v7_t;

typedef struct rcclNetProperties_v7 {
  char name[MAX_STR_LEN];
  char pciPath[MAX_STR_LEN];
  uint64_t guid;
  int ptrSupport;
  int speed;
  int port;
  int maxComm;
  int maxRecvs;
} rcclNetProperties_v7_t;

#endif // RCCL_NET_V7_H_
