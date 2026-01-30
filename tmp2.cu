#include <iostream>
#include "cuda_runtime.h"
int main()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // 0 - номер устройства

    printf("Name of GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Max shared memory per block (KB): %zu\n", prop.sharedMemPerBlock / 1024);
    return 0;
}