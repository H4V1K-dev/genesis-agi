#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

// Дублируем контракт памяти из bindings.cu
struct ShardVramPtrs {
  int32_t *soma_voltage;
  uint8_t *soma_flags;
  int32_t *threshold_offset;
  uint8_t *timers;
  uint32_t *soma_to_axon;
  uint32_t *dendrite_targets;
  int16_t *dendrite_weights;
  uint32_t *axon_heads;
};

#define AXON_SENTINEL 0x80000000
#define MAX_DENDRITES 128

// Строго 64 байта. 16 типов = 1024 байта (идеально ложится в кеш L1 constant)
struct VariantParameters {
  int32_t threshold;
  int32_t rest_potential;
  int32_t leak_rate;
  int32_t homeostasis_penalty;
  int32_t homeostasis_decay;
  int32_t gsop_potentiation;
  int32_t gsop_depression;
  uint8_t refractory_period;
  uint8_t synapse_refractory_period;
  uint8_t slot_decay_ltm;
  uint8_t slot_decay_wm;
  uint8_t signal_propagation_length;
  uint8_t ltm_slot_count;
  uint8_t _pad1[2];          // Выравнивание до 36B
  int16_t inertia_curve[16]; // 32B — кривая инерции GSOP (16 рангов)
  uint8_t _pad2[60];         // Дополняем до 128 байт
};

// Глобальная константная память. Rust будет заливать сюда конфиг перед стартом.
__constant__ VariantParameters VARIANT_LUT[16];
extern __constant__ int16_t current_dopamine;

// ============================================================================
// 1. Inject Inputs Kernel (Virtual Axons)
// Извлекает биты из плотной маски и сбрасывает головы виртуальным аксонам
// ============================================================================
__global__ void cu_inject_inputs_kernel(uint32_t *axon_heads,
                                        const uint32_t *input_bitmask,
                                        uint32_t virtual_offset,
                                        uint32_t num_virtual_axons) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_virtual_axons)
    return;

  // Извлечение бита за 2 такта ALU (деление на 32 компилятор оптимизирует в
  // shift)
  uint32_t word_idx = tid / 32;
  uint32_t bit_idx = tid % 32;
  bool is_active = (input_bitmask[word_idx] >> bit_idx) & 1;

  // Ветвление минимизировано: пишем только если есть пульс
  if (is_active) {
    axon_heads[virtual_offset + tid] = 0; // Рождение сигнала
  }
}

// ============================================================================
// 2. Apply Spike Batch Kernel (Network / Ghost Axons)
// O(1) инъекция сетевых спайков через Sender-Side Mapping
// ============================================================================
__global__ void cu_apply_spike_batch_kernel(uint32_t *axon_heads,
                                            const uint32_t *incoming_spikes,
                                            uint32_t num_incoming_spikes) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_incoming_spikes)
    return;

  // Sender-Side Mapping гарантирует, что incoming_spikes[tid] — это готовый
  // локальный индекс в массиве axon_heads. Никаких трансляций ID.
  uint32_t ghost_id = incoming_spikes[tid];

  // Сброс головы Ghost-аксона в 0 (рождение сигнала от соседа)
  axon_heads[ghost_id] = 0;
}

// ============================================================================
// 3. Propagate Axons Kernel
// ============================================================================
__global__ void cu_propagate_axons_kernel(uint32_t *axon_heads,
                                          uint32_t total_axons,
                                          uint32_t v_seg) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total_axons)
    return;

  uint32_t head = axon_heads[tid];
  if (head != AXON_SENTINEL) {
    axon_heads[tid] = head + v_seg;
  }
}

// ============================================================================
// 4. Update Neurons Kernel (GLIF + Dendritic Integration)
// ============================================================================
__global__ void cu_update_neurons_kernel(ShardVramPtrs vram,
                                         uint32_t padded_n) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= padded_n)
    return;

  uint8_t flags = vram.soma_flags[tid];
  uint8_t timer = vram.timers[tid];

  flags &= ~0x01;

  if (timer > 0) {
    vram.timers[tid] = timer - 1;
    vram.soma_flags[tid] = flags;
    return;
  }

  uint8_t variant_id = (flags >> 4) & 0x0F;
  VariantParameters p = VARIANT_LUT[variant_id];

  int32_t current_voltage = vram.soma_voltage[tid];
  int32_t i_in = 0;

  for (int i = 0; i < MAX_DENDRITES; i++) {
    uint32_t col_idx = i * padded_n + tid;
    uint32_t target_packed = vram.dendrite_targets[col_idx];

    if (target_packed == 0)
      continue;

    uint32_t target_id = target_packed & 0x00FFFFFF;
    uint32_t seg_idx = target_packed >> 24;

    uint32_t head = vram.axon_heads[target_id];

    uint32_t dist = head - seg_idx;
    if (dist < p.signal_propagation_length) {
      i_in += (int32_t)vram.dendrite_weights[col_idx];
    }
  }

  int32_t thresh_offset = vram.threshold_offset[tid];
  if (thresh_offset > 0) {
    thresh_offset -= p.homeostasis_decay;
    if (thresh_offset < 0)
      thresh_offset = 0;
  }

  current_voltage += i_in - p.leak_rate;
  int32_t effective_threshold = p.threshold + thresh_offset;

  if (current_voltage >= effective_threshold) {
    flags |= 0x01;
    current_voltage = p.rest_potential;
    thresh_offset += p.homeostasis_penalty;
    vram.timers[tid] = p.refractory_period;

    uint32_t my_axon = vram.soma_to_axon[tid];
    if (my_axon != 0xFFFFFFFF) {
      vram.axon_heads[my_axon] = 0;
    }
  }

  vram.soma_voltage[tid] = current_voltage;
  vram.soma_flags[tid] = flags;
  vram.threshold_offset[tid] = thresh_offset;
}

// ============================================================================
// 5. Apply GSOP Kernel (Spatial STDP Plasticity)
// ============================================================================
__global__ void cu_apply_gsop_kernel(ShardVramPtrs vram, uint32_t padded_n) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= padded_n)
    return;

  uint8_t flags = vram.soma_flags[tid];
  if ((flags & 0x01) == 0)
    return;

  uint8_t variant_id = (flags >> 4) & 0x0F;
  VariantParameters p = VARIANT_LUT[variant_id];

  for (int i = 0; i < MAX_DENDRITES; i++) {
    uint32_t col_idx = i * padded_n + tid;
    uint32_t target_packed = vram.dendrite_targets[col_idx];

    if (target_packed == 0)
      break; // Пустые слоты в хвосте

    uint32_t target_id = target_packed & 0x00FFFFFF;
    uint32_t seg_idx = target_packed >> 24;
    uint32_t head = vram.axon_heads[target_id];

    uint32_t dist = head - seg_idx;
    bool is_active = (dist < p.signal_propagation_length);

    int16_t w = vram.dendrite_weights[col_idx];
    int16_t sign = (w >= 0) ? 1 : -1;
    int32_t abs_w = (int32_t)w;
    if (abs_w < 0)
      abs_w = -abs_w;

    // 1. Inertia Rank (1 такт, Branchless)
    uint32_t rank = abs_w >> 11;
    if (rank > 15)
      rank = 15;
    int32_t inertia = p.inertia_curve[rank];

    // 2. Modulated Potentiation / Depression
    int32_t base_pot = p.gsop_potentiation;
    int32_t dopa_mod = (base_pot * current_dopamine) >> 8;
    int32_t final_pot = base_pot + dopa_mod;

    int32_t delta_pot = (final_pot * inertia) >> 7;
    int32_t delta_dep = (p.gsop_depression * inertia) >> 7;

    // 3. Causal Delta
    int32_t delta = is_active ? delta_pot : -delta_dep;

    // 4. Slot Decay (масштабирует дельту, а не вычитается из веса!)
    int32_t decay = (i < p.ltm_slot_count) ? p.slot_decay_ltm : p.slot_decay_wm;
    delta = (delta * decay) >> 7;

    // 5. Apply & Clamp
    int32_t new_abs = abs_w + delta;
    if (new_abs < 0)
      new_abs = 0;
    if (new_abs > 32767)
      new_abs = 32767;

    vram.dendrite_weights[col_idx] = (int16_t)(new_abs * sign);
  }
}

// ============================================================================
// 6. Record Readout Kernel (Output Matrix)
// ============================================================================
__global__ void cu_record_readout_kernel(const uint8_t *soma_flags,
                                         const uint32_t *mapped_soma_ids,
                                         uint8_t *output_history,
                                         uint32_t num_outputs) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_outputs)
    return;

  uint32_t soma_id = mapped_soma_ids[tid];
  uint8_t is_spiking = 0;

  // [DOD] Защита от Memory Out-of-Bounds. Сентинел означает пустой пиксель.
  if (soma_id != 0xFFFFFFFF) {
    is_spiking = soma_flags[soma_id] & 0x01;
  }

  output_history[tid] = is_spiking;
}

extern "C" {

// ============================================================================
// Day Phase Orchestrator
// ============================================================================
int32_t cu_step_day_phase(const ShardVramPtrs *vram, uint32_t padded_n,
                          uint32_t total_axons, uint32_t v_seg,
                          // --- ВХОДЫ (InjectInputs) ---
                          const uint32_t *input_bitmask,
                          uint32_t virtual_offset, uint32_t num_virtual_axons,
                          // --- СЕТЬ (ApplySpikeBatch) ---
                          const uint32_t *incoming_spikes,
                          uint32_t num_incoming_spikes,
                          // --- ВЫХОДЫ (RecordReadout) ---
                          const uint32_t *mapped_soma_ids,
                          uint8_t *output_history, uint32_t num_outputs) {
  int threads = 256;

  // 1. InjectInputs (Только если есть виртуальные аксоны и передана маска)
  if (num_virtual_axons > 0 && input_bitmask != nullptr) {
    int blocks_in = (num_virtual_axons + threads - 1) / threads;
    cu_inject_inputs_kernel<<<blocks_in, threads>>>(
        vram->axon_heads, input_bitmask, virtual_offset, num_virtual_axons);
  }

  // 2. ApplySpikeBatch (Сетевые спайки от соседних зон)
  if (num_incoming_spikes > 0 && incoming_spikes != nullptr) {
    int blocks_spikes = (num_incoming_spikes + threads - 1) / threads;
    cu_apply_spike_batch_kernel<<<blocks_spikes, threads>>>(
        vram->axon_heads, incoming_spikes, num_incoming_spikes);
  }

  // 3. PropagateAxons
  int blocks_axons = (total_axons + threads - 1) / threads;
  cu_propagate_axons_kernel<<<blocks_axons, threads>>>(vram->axon_heads,
                                                       total_axons, v_seg);

  // 4. UpdateNeurons (GLIF)
  int blocks_neurons = (padded_n + threads - 1) / threads;
  cu_update_neurons_kernel<<<blocks_neurons, threads>>>(*vram, padded_n);

  // 5. ApplyGSOP (Пластичность 3D STDP)
  cu_apply_gsop_kernel<<<blocks_neurons, threads>>>(*vram, padded_n);

  // 6. RecordReadout
  if (num_outputs > 0 && mapped_soma_ids != nullptr &&
      output_history != nullptr) {
    int blocks_out = (num_outputs + threads - 1) / threads;
    cu_record_readout_kernel<<<blocks_out, threads>>>(
        vram->soma_flags, mapped_soma_ids, output_history, num_outputs);
  }

  return 0;
}

// Позволяет заливать параметры вариантов в константную память GPU
int32_t cu_upload_constant_memory(const VariantParameters *lut) {
  return cudaMemcpyToSymbol(VARIANT_LUT, lut, sizeof(VariantParameters) * 16);
}

} // extern "C"
