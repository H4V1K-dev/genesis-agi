#include <cuda_runtime.h>
#include <stdint.h>

extern "C" void launch_apply_spike_batch(uint32_t current_tick,
                                         uint32_t num_spikes_this_tick,
                                         uint32_t *schedule_indices,
                                         uint32_t *axon_heads, void *stream) {
  if (num_spikes_this_tick == 0)
    return;

  // Fast path zero-copy injection: Schedule indices are exactly the ghost IDs.
  // We launch exactly one thread per spike.
  int blockSize = 128; // Standard block size for this low-workbound kernel
  int numBlocks = (num_spikes_this_tick + blockSize - 1) / blockSize;

  // Lambda wrapper or direct kernel:
  // We place the kernel logic inline using a lambda if we wanted, but standard
  // __global__ is better.
}

__global__ void apply_spike_batch_kernel(uint32_t num_spikes,
                                         uint32_t *schedule_indices,
                                         uint32_t *axon_heads) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < num_spikes) {
    // Direct receiver Ghost ID from the O(1) sender-mapped array
    uint32_t ghost_id = schedule_indices[tid];

    // Zero out the axon head to simulate a fresh spike arriving at the physical
    // layer
    axon_heads[ghost_id] = 0;
  }
}

// Redefine the ffi function implementation properly
extern "C" void launch_apply_spike_batch_impl(uint32_t num_spikes,
                                              uint32_t *schedule_indices,
                                              uint32_t *axon_heads,
                                              void *stream) {
  if (num_spikes == 0)
    return;
  int blockSize = 128;
  int numBlocks = (num_spikes + blockSize - 1) / blockSize;
  apply_spike_batch_kernel<<<numBlocks, blockSize, 0, (cudaStream_t)stream>>>(
      num_spikes, schedule_indices, axon_heads);
}
