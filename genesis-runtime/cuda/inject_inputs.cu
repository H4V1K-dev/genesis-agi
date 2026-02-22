#include <cuda_runtime.h>
#include <stdint.h>

// 05_signal_physics.md §2.4 Kernel (InjectInputs)
__global__ void inject_inputs_kernel(uint32_t *axon_heads,
                                     const uint32_t *input_bitmask,
                                     uint32_t virtual_offset,
                                     uint32_t num_virtual) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_virtual)
    return;

  // Broadcast read: 32 потока варпа → 1 u32 из bitmask
  uint32_t mask = input_bitmask[tid / 32];
  uint32_t is_active = (mask >> (tid % 32)) & 1;

  // Write-Only: if выгоднее branchless (избегаем global read)
  // Рождение сигнала = сброс axon_heads[id] = 0
  if (is_active) {
    axon_heads[virtual_offset + tid] = 0;
  }
}

extern "C" void launch_inject_inputs(uint32_t *axon_heads,
                                     const uint32_t *input_bitmask,
                                     uint32_t virtual_offset,
                                     uint32_t num_virtual, void *stream) {
  int blockSize = 256;
  int numBlocks = (num_virtual + blockSize - 1) / blockSize;

  // Ensure we don't divide by zero if num_virtual is 0
  if (numBlocks > 0) {
    inject_inputs_kernel<<<numBlocks, blockSize, 0, (cudaStream_t)stream>>>(
        axon_heads, input_bitmask, virtual_offset, num_virtual);
  }
}
