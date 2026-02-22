#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

extern "C" {

// ---- Basic Memory Ops ----

void* gpu_malloc(size_t size) {
    void* dev_ptr = nullptr;
    cudaError_t err = cudaMalloc(&dev_ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Malloc failed: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    return dev_ptr;
}

void gpu_free(void* dev_ptr) {
    if (dev_ptr) {
        cudaFree(dev_ptr);
    }
}

bool gpu_memcpy_host_to_device(void* dst_dev, const void* src_host, size_t size) {
    cudaError_t err = cudaMemcpy(dst_dev, src_host, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Memcpy H2D failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

bool gpu_memcpy_device_to_host(void* dst_host, const void* src_dev, size_t size) {
    cudaError_t err = cudaMemcpy(dst_host, src_dev, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Memcpy D2H failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

// Synchronizes the device, waits for all streams.
void gpu_device_synchronize() {
    cudaDeviceSynchronize();
}

} // extern "C"
