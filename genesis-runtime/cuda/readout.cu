#include <cuda_runtime.h>
#include <stdint.h>

__global__ void record_readout_kernel(
    const uint8_t* flags,
    const uint32_t* mapped_soma_ids,
    uint8_t* output_history,
    uint32_t total_mapped_somas,
    uint32_t current_tick_in_batch
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_mapped_somas) return;

    uint32_t soma_id = mapped_soma_ids[tid];
    uint8_t is_spiking = flags[soma_id] & 0x01;

    // Coalesced write: threads 0..31 write consecutive bytes
    output_history[current_tick_in_batch * total_mapped_somas + tid] = is_spiking;
}

extern "C" void launch_record_readout(
    const void* flags,
    const void* mapped_soma_ids,
    void* output_history,
    uint32_t total_mapped_somas,
    uint32_t current_tick_in_batch,
    void* stream
) {
    if (total_mapped_somas == 0) return;

    int blockSize = 128; // Standard optimization block size
    int numBlocks = (total_mapped_somas + blockSize - 1) / blockSize;

    record_readout_kernel<<<numBlocks, blockSize, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)flags,
        (const uint32_t*)mapped_soma_ids,
        (uint8_t*)output_history,
        total_mapped_somas,
        current_tick_in_batch
    );
}
