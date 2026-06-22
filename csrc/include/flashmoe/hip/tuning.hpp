//
// FlashMoE HIP Port - Performance Tuning Configurations
// Optimized for AMD Instinct MI450 (gfx1250) / MI450 (gfx1250)
//
// Copyright (c) 2025, Osayamen Jonathan Aimuyo
// All rights reserved.
//

#ifndef FLASHMOE_HIP_TUNING_HPP
#define FLASHMOE_HIP_TUNING_HPP

#include <hip/hip_runtime.h>
#include <cstdint>
#include <algorithm>
#include <limits>
#include "constants.hpp"

namespace flashmoe::tuning {

// ============================================================================
// Section 1: MI450 MFMA Instruction Configuration
// ============================================================================

namespace mfma {

// MFMA instruction sizes supported on gfx1250 (MI450/MI450)
// Format: MFMA_{output_type}_{M}x{N}x{K}_{input_type}

// 16x16x16 MFMA for FP16/BF16 (higher K, better for small tiles)
struct MFMA_16x16x16 {
    static constexpr int M = 16;
    static constexpr int N = 16;
    static constexpr int K = 16;
    static constexpr int ops_per_instruction = M * N * K * 2;  // FMA = 2 ops
    static constexpr int cycles = 8;  // Approximate cycles per MFMA
    static constexpr double ops_per_cycle = ops_per_instruction / static_cast<double>(cycles);
};

// 32x32x8 MFMA for FP16/BF16 (larger tile, better for big matrices)
struct MFMA_32x32x8 {
    static constexpr int M = 32;
    static constexpr int N = 32;
    static constexpr int K = 8;
    static constexpr int ops_per_instruction = M * N * K * 2;
    static constexpr int cycles = 16;
    static constexpr double ops_per_cycle = ops_per_instruction / static_cast<double>(cycles);
};

// 32x32x4 MFMA for FP32 (TF32-like)
struct MFMA_32x32x4_FP32 {
    static constexpr int M = 32;
    static constexpr int N = 32;
    static constexpr int K = 4;
    static constexpr int ops_per_instruction = M * N * K * 2;
    static constexpr int cycles = 32;
    static constexpr double ops_per_cycle = ops_per_instruction / static_cast<double>(cycles);
};

// 16x16x4 MFMA for FP32
struct MFMA_16x16x4_FP32 {
    static constexpr int M = 16;
    static constexpr int N = 16;
    static constexpr int K = 4;
    static constexpr int ops_per_instruction = M * N * K * 2;
    static constexpr int cycles = 8;
    static constexpr double ops_per_cycle = ops_per_instruction / static_cast<double>(cycles);
};

} // namespace mfma

// ============================================================================
// Section 2: Tile Size Configurations
// ============================================================================

namespace tile {

// Base tile configuration struct
template<int BlockM, int BlockN, int BlockK,
         int WmmaM, int WmmaN, int WmmaK,
         int Threads, int PipeStages>
struct TileConfig {
    // Block dimensions
    static constexpr int bM = BlockM;
    static constexpr int bN = BlockN;
    static constexpr int bK = BlockK;

    // WMMA tile dimensions
    static constexpr int wmma_m = WmmaM;
    static constexpr int wmma_n = WmmaN;
    static constexpr int wmma_k = WmmaK;

    // Execution parameters
    static constexpr int threads = Threads;
    static constexpr int pipe_stages = PipeStages;

    // Derived dimensions
    static constexpr int tiles_m = bM / wmma_m;
    static constexpr int tiles_n = bN / wmma_n;
    static constexpr int tiles_k = bK / wmma_k;

    // Wavefront count
    static constexpr int waves = threads / flashmoe::WAVEFRONT_SIZE;

    // LDS requirements (in bytes, assuming FP16)
    static constexpr int lds_a_per_stage = bM * bK * 2;  // 2 bytes for FP16
    static constexpr int lds_b_per_stage = bK * bN * 2;
    static constexpr int lds_per_stage = lds_a_per_stage + lds_b_per_stage;
    static constexpr int lds_total = lds_per_stage * pipe_stages;

    // Validation: LDS must fit in 64KB
    static_assert(lds_total <= 65536, "LDS requirements exceed 64KB limit");

    // Compute intensity (ops per byte loaded)
    static constexpr double compute_intensity =
        static_cast<double>(bM * bN * bK * 2) /
        static_cast<double>((bM * bK + bK * bN) * 2);  // FP16 = 2 bytes
};

// ============================================================================
// Optimal Tile Configurations for Different GEMM Shapes
// ============================================================================

// Small tiles (for small M, N dimensions or high occupancy)
using TileSmall_32x32x32 = TileConfig<32, 32, 32, 16, 16, 16, 128, 2>;
using TileSmall_32x64x32 = TileConfig<32, 64, 32, 16, 16, 16, 128, 2>;
using TileSmall_64x32x32 = TileConfig<64, 32, 32, 16, 16, 16, 128, 2>;

// Medium tiles (balanced compute and memory)
using TileMedium_64x64x32 = TileConfig<64, 64, 32, 16, 16, 16, 256, 2>;
using TileMedium_64x64x64 = TileConfig<64, 64, 64, 16, 16, 16, 256, 2>;
using TileMedium_64x128x32 = TileConfig<64, 128, 32, 16, 16, 16, 256, 2>;
using TileMedium_128x64x32 = TileConfig<128, 64, 32, 16, 16, 16, 256, 2>;

// Large tiles (for large matrices, high compute intensity)
using TileLarge_128x128x32 = TileConfig<128, 128, 32, 32, 32, 8, 256, 2>;
using TileLarge_128x128x64 = TileConfig<128, 128, 64, 32, 32, 8, 256, 3>;
using TileLarge_128x256x32 = TileConfig<128, 256, 32, 32, 32, 8, 256, 2>;
using TileLarge_256x128x32 = TileConfig<256, 128, 32, 32, 32, 8, 256, 2>;

// Extra large tiles (maximum compute intensity, lower occupancy)
using TileXLarge_256x256x32 = TileConfig<256, 256, 32, 32, 32, 8, 512, 2>;
using TileXLarge_256x256x64 = TileConfig<256, 256, 64, 32, 32, 8, 512, 2>;

// ============================================================================
// Runtime Tile Selection Structure
// ============================================================================

struct TileParams {
    int bM;
    int bN;
    int bK;
    int wmma_m;
    int wmma_n;
    int wmma_k;
    int threads;
    int pipe_stages;
    int lds_bytes;
    double compute_intensity;
    const char* name;
};

// Predefined configurations array for runtime selection
inline constexpr TileParams TILE_CONFIGS[] = {
    // Small tiles
    {32, 32, 32, 16, 16, 16, 128, 2, 8192, 8.0, "Small_32x32x32"},
    {32, 64, 32, 16, 16, 16, 128, 2, 12288, 10.67, "Small_32x64x32"},
    {64, 32, 32, 16, 16, 16, 128, 2, 12288, 10.67, "Small_64x32x32"},

    // Medium tiles
    {64, 64, 32, 16, 16, 16, 256, 2, 16384, 16.0, "Medium_64x64x32"},
    {64, 64, 64, 16, 16, 16, 256, 2, 32768, 16.0, "Medium_64x64x64"},
    {64, 128, 32, 16, 16, 16, 256, 2, 24576, 21.33, "Medium_64x128x32"},
    {128, 64, 32, 16, 16, 16, 256, 2, 24576, 21.33, "Medium_128x64x32"},

    // Large tiles
    {128, 128, 32, 32, 32, 8, 256, 2, 32768, 32.0, "Large_128x128x32"},
    {128, 128, 64, 32, 32, 8, 256, 3, 98304, 32.0, "Large_128x128x64"},
    {128, 256, 32, 32, 32, 8, 256, 2, 49152, 42.67, "Large_128x256x32"},
    {256, 128, 32, 32, 32, 8, 256, 2, 49152, 42.67, "Large_256x128x32"},

    // Extra large tiles
    {256, 256, 32, 32, 32, 8, 512, 2, 65536, 64.0, "XLarge_256x256x32"},
};

inline constexpr int NUM_TILE_CONFIGS = sizeof(TILE_CONFIGS) / sizeof(TILE_CONFIGS[0]);

} // namespace tile

// ============================================================================
// Section 3: Occupancy Configurations
// ============================================================================

namespace occupancy {

// Occupancy tuning placeholders.
constexpr int TOTAL_SCHEDULING_UNITS = 1;
constexpr int MAX_ACTIVE_WAVES_REFERENCE = 1;
constexpr int VGPR_BUDGET_REFERENCE = 1;
constexpr int SGPR_BUDGET_REFERENCE = 1;
constexpr int MAX_LDS_PER_WORKGROUP = flashmoe::MAX_LDS_BYTES;

// Calculate maximum waves per SIMD based on VGPR usage
__host__ __device__ constexpr int calc_waves_per_simd(int vgprs_per_wave) {
    int waves = VGPR_BUDGET_REFERENCE / vgprs_per_wave;
    return (waves > MAX_ACTIVE_WAVES_REFERENCE) ? MAX_ACTIVE_WAVES_REFERENCE : waves;
}

// Calculate theoretical occupancy (0.0 to 1.0)
__host__ __device__ constexpr double calc_occupancy(int threads_per_block,
                                                     int vgprs_per_thread,
                                                     int lds_bytes) {
    int waves_per_block = threads_per_block / flashmoe::WAVEFRONT_SIZE;
    int vgprs_per_wave = vgprs_per_thread * flashmoe::WAVEFRONT_SIZE;

    // VGPR limit
    int waves_vgpr_limited = calc_waves_per_simd(vgprs_per_wave);

    // LDS limit (one workgroup per CU if using max LDS)
    int blocks_lds_limited = MAX_LDS_PER_WORKGROUP / ((lds_bytes > 0) ? lds_bytes : 1);

    // Effective waves per SIMD
    int effective_waves = (waves_vgpr_limited * blocks_lds_limited * waves_per_block);
    if (effective_waves > MAX_ACTIVE_WAVES_REFERENCE) effective_waves = MAX_ACTIVE_WAVES_REFERENCE;

    return static_cast<double>(effective_waves) / MAX_ACTIVE_WAVES_REFERENCE;
}

// Occupancy configuration presets
struct OccupancyConfig {
    int vgprs_per_thread;
    int lds_bytes;
    int threads_per_block;
    double theoretical_occupancy;
    const char* description;
};

inline constexpr OccupancyConfig OCCUPANCY_PRESETS[] = {
    // High occupancy (smaller tiles, more waves)
    {8, 8192, 128, 1.0, "High occupancy - 8 VGPRs"},
    {16, 16384, 256, 1.0, "High occupancy - 16 VGPRs"},

    // Balanced occupancy
    {32, 32768, 256, 0.5, "Balanced - 32 VGPRs"},
    {48, 49152, 256, 0.33, "Balanced - 48 VGPRs"},

    // Low occupancy (large tiles, high ILP)
    {64, 65536, 256, 0.25, "Low occupancy - 64 VGPRs"},
    {96, 65536, 512, 0.167, "Low occupancy - 96 VGPRs"},
};

// Calculate optimal thread count for given problem size
inline int suggest_threads(int M, int N, int K, int bM, int bN) {
    int total_tiles = ((M + bM - 1) / bM) * ((N + bN - 1) / bN);

    // For small problems, use fewer threads for higher occupancy
    if (total_tiles < TOTAL_SCHEDULING_UNITS) {
        return 256;  // 4 waves
    }

    // For large problems, can use more threads per block
    if (M >= 4096 && N >= 4096) {
        return 512;  // 8 waves
    }

    return 256;  // Default: 4 waves
}

} // namespace occupancy

// ============================================================================
// Section 4: Memory Access Patterns
// ============================================================================

namespace memory {

constexpr int L2_CACHE_LINE_SIZE = 128;  // bytes

// Optimal vectorization widths (128-bit loads)
constexpr int VECTOR_WIDTH_BYTES = 16;  // 128 bits
constexpr int VECTOR_WIDTH_FP16 = VECTOR_WIDTH_BYTES / 2;   // 8 elements
constexpr int VECTOR_WIDTH_FP32 = VECTOR_WIDTH_BYTES / 4;   // 4 elements
constexpr int VECTOR_WIDTH_BF16 = VECTOR_WIDTH_BYTES / 2;   // 8 elements

// Memory access configuration
struct MemoryConfig {
    int vector_width;      // Elements per vector load
    int prefetch_distance; // Cache lines to prefetch ahead
    bool use_async_copy;   // Use LDS staging
    int coalesce_width;    // Threads for coalesced access
};

// Optimal memory access configurations
inline constexpr MemoryConfig MEM_CONFIG_FP16 = {
    .vector_width = 8,
    .prefetch_distance = 4,
    .use_async_copy = true,
    .coalesce_width = flashmoe::WARP_SIZE
};

inline constexpr MemoryConfig MEM_CONFIG_FP32 = {
    .vector_width = 4,
    .prefetch_distance = 4,
    .use_async_copy = true,
    .coalesce_width = flashmoe::WARP_SIZE
};

inline constexpr MemoryConfig MEM_CONFIG_BF16 = {
    .vector_width = 8,
    .prefetch_distance = 4,
    .use_async_copy = true,
    .coalesce_width = 64
};

// Calculate memory bandwidth requirement
inline double calc_bandwidth_requirement(int M, int N, int K,
                                         int element_bytes,
                                         double target_tflops) {
    // GEMM ops = 2 * M * N * K
    double total_ops = 2.0 * M * N * K;

    // Bytes to load: A (M*K) + B (K*N), write: C (M*N)
    double bytes_loaded = static_cast<double>((M * K + K * N + M * N) * element_bytes);

    // Time for target TFLOPS
    double time_seconds = total_ops / (target_tflops * 1e12);

    // Required bandwidth
    return bytes_loaded / time_seconds / 1e12;  // TB/s
}

// Check if problem is memory-bound
inline bool is_memory_bound(int M, int N, int K, int element_bytes) {
    double arithmetic_intensity = static_cast<double>(2 * M * N * K) /
                                  ((M * K + K * N + M * N) * element_bytes);

    return arithmetic_intensity < 16.0;
}

} // namespace memory

// ============================================================================
// Section 5: Pipeline Configurations
// ============================================================================

namespace pipeline {

// Pipeline stage configuration
struct PipelineConfig {
    int stages;            // Number of pipeline stages (2-4)
    int prefetch_depth;    // How far ahead to prefetch
    bool double_buffer;    // Use double buffering for LDS
    bool async_copy;       // Use async copy (simulated on AMD)
    int compute_overlap;   // Compute stages overlapping with memory
};

// Optimal pipeline configurations based on problem characteristics
inline constexpr PipelineConfig PIPELINE_SMALL = {
    .stages = 2,
    .prefetch_depth = 1,
    .double_buffer = true,
    .async_copy = false,
    .compute_overlap = 1
};

inline constexpr PipelineConfig PIPELINE_MEDIUM = {
    .stages = 2,
    .prefetch_depth = 2,
    .double_buffer = true,
    .async_copy = true,
    .compute_overlap = 1
};

inline constexpr PipelineConfig PIPELINE_LARGE = {
    .stages = 3,
    .prefetch_depth = 2,
    .double_buffer = true,
    .async_copy = true,
    .compute_overlap = 2
};

inline constexpr PipelineConfig PIPELINE_XLARGE = {
    .stages = 4,
    .prefetch_depth = 3,
    .double_buffer = true,
    .async_copy = true,
    .compute_overlap = 2
};

// Select pipeline configuration based on K dimension
inline const PipelineConfig& select_pipeline(int K, int bK) {
    int k_tiles = K / bK;

    if (k_tiles <= 4) {
        return PIPELINE_SMALL;
    } else if (k_tiles <= 16) {
        return PIPELINE_MEDIUM;
    } else if (k_tiles <= 64) {
        return PIPELINE_LARGE;
    } else {
        return PIPELINE_XLARGE;
    }
}

// Calculate LDS requirement for pipeline
inline int calc_pipeline_lds(int bM, int bN, int bK, int stages, int element_bytes) {
    int lds_a = bM * bK * element_bytes;
    int lds_b = bK * bN * element_bytes;
    return (lds_a + lds_b) * stages;
}

} // namespace pipeline

// ============================================================================
// Section 6: MoE Expert Size Configurations
// ============================================================================

namespace moe {

// Common MoE hidden dimension configurations
enum class ExpertSize {
    SMALL,      // hidden_dim = 256-512
    MEDIUM,     // hidden_dim = 1024-2048
    LARGE,      // hidden_dim = 4096+
    XLARGE      // hidden_dim = 8192+
};

// Expert configuration template
template<ExpertSize Size>
struct ExpertConfig;

// Small experts (256-512 hidden dim)
template<>
struct ExpertConfig<ExpertSize::SMALL> {
    // Typical dimensions
    static constexpr int min_hidden = 256;
    static constexpr int max_hidden = 512;

    // Recommended tile configuration
    static constexpr int bM = 32;
    static constexpr int bN = 64;
    static constexpr int bK = 32;

    // WMMA size (use smaller for better utilization)
    static constexpr int wmma_m = 16;
    static constexpr int wmma_n = 16;
    static constexpr int wmma_k = 16;

    // Execution parameters
    static constexpr int threads = 128;
    static constexpr int pipe_stages = 2;

    // Optimize for high occupancy
    static constexpr bool high_occupancy = true;
    static constexpr int target_waves_per_simd = 4;
};

// Medium experts (1024-2048 hidden dim)
template<>
struct ExpertConfig<ExpertSize::MEDIUM> {
    static constexpr int min_hidden = 1024;
    static constexpr int max_hidden = 2048;

    static constexpr int bM = 64;
    static constexpr int bN = 64;
    static constexpr int bK = 32;

    static constexpr int wmma_m = 16;
    static constexpr int wmma_n = 16;
    static constexpr int wmma_k = 16;

    static constexpr int threads = 256;
    static constexpr int pipe_stages = 2;

    static constexpr bool high_occupancy = true;
    static constexpr int target_waves_per_simd = 2;
};

// Large experts (4096+ hidden dim)
template<>
struct ExpertConfig<ExpertSize::LARGE> {
    static constexpr int min_hidden = 4096;
    static constexpr int max_hidden = 8191;

    static constexpr int bM = 128;
    static constexpr int bN = 128;
    static constexpr int bK = 32;

    static constexpr int wmma_m = 32;
    static constexpr int wmma_n = 32;
    static constexpr int wmma_k = 8;

    static constexpr int threads = 256;
    static constexpr int pipe_stages = 2;

    static constexpr bool high_occupancy = false;
    static constexpr int target_waves_per_simd = 1;
};

// Extra large experts (8192+ hidden dim)
template<>
struct ExpertConfig<ExpertSize::XLARGE> {
    static constexpr int min_hidden = 8192;
    static constexpr int max_hidden = 32768;

    static constexpr int bM = 256;
    static constexpr int bN = 256;
    static constexpr int bK = 32;

    static constexpr int wmma_m = 32;
    static constexpr int wmma_n = 32;
    static constexpr int wmma_k = 8;

    static constexpr int threads = 512;
    static constexpr int pipe_stages = 2;

    static constexpr bool high_occupancy = false;
    static constexpr int target_waves_per_simd = 1;
};

// Runtime expert size detection
inline ExpertSize detect_expert_size(int hidden_dim) {
    if (hidden_dim <= 512) return ExpertSize::SMALL;
    if (hidden_dim <= 2048) return ExpertSize::MEDIUM;
    if (hidden_dim <= 8191) return ExpertSize::LARGE;
    return ExpertSize::XLARGE;
}

// Runtime configuration lookup
struct ExpertParams {
    int bM, bN, bK;
    int wmma_m, wmma_n, wmma_k;
    int threads;
    int pipe_stages;
    bool high_occupancy;
};

// Heuristic tile selection:
// - Smaller tiles improve occupancy on small expert batches.
// - 16x16x16 WMMA is a conservative default for medium sizes.
// - Deeper K tiles can improve reuse for larger problems.
inline ExpertParams get_expert_params(int hidden_dim) {
    ExpertSize size = detect_expert_size(hidden_dim);

    switch (size) {
        case ExpertSize::SMALL:
            // Small experts: use smallest tiles for occupancy
            return {32, 32, 32, 16, 16, 16, 128, 2, true};
        case ExpertSize::MEDIUM:
            // Medium experts: 64x64 tiles work well
            return {64, 64, 32, 16, 16, 16, 256, 2, true};
        case ExpertSize::LARGE:
            // Large experts: 64x64 tiles still competitive
            return {64, 64, 32, 16, 16, 16, 256, 2, false};
        case ExpertSize::XLARGE:
        default:
            // XLarge experts: use larger tiles with deeper K
            return {128, 128, 64, 32, 32, 8, 256, 3, false};
    }
}

} // namespace moe

// ============================================================================
// Section 7: Autotuning Hints and Selection Logic
// ============================================================================

namespace autotune {

// Performance model estimate (relative score, higher is better)
struct PerfEstimate {
    double compute_score;     // Based on MFMA utilization
    double memory_score;      // Based on bandwidth utilization
    double occupancy_score;   // Based on CU occupancy
    double combined_score;    // Weighted combination
};

// Estimate performance for a tile configuration
inline PerfEstimate estimate_performance(int M, int N, int K,
                                         const tile::TileParams& config,
                                         bool is_fp16 = true) {
    // Compute score: larger tiles with good WMMA utilization
    double tile_efficiency = static_cast<double>(config.bM * config.bN) / (256.0 * 256.0);
    double wmma_utilization = (config.wmma_m == 32) ? 1.0 : 0.8;  // 32x32 MFMA is slightly better
    double compute_score = tile_efficiency * wmma_utilization;

    // Memory score: based on compute intensity (ops per byte)
    double peak_intensity = 64.0;  // Approximate roofline intersection
    double memory_score = std::min(1.0, config.compute_intensity / peak_intensity);

    // Occupancy score: based on LDS and thread count
    double lds_occupancy = 1.0 - (static_cast<double>(config.lds_bytes) / 65536.0);
    double thread_occupancy = static_cast<double>(config.threads) / 1024.0;
    double occupancy_score = (lds_occupancy + thread_occupancy) / 2.0;

    // Problem size adjustments
    int total_tiles = ((M + config.bM - 1) / config.bM) *
                     ((N + config.bN - 1) / config.bN);

    // Penalize configurations with too few tiles (poor CU utilization)
    double utilization_penalty = (total_tiles < occupancy::TOTAL_SCHEDULING_UNITS) ?
                                 (static_cast<double>(total_tiles) / occupancy::TOTAL_SCHEDULING_UNITS) : 1.0;

    // Combined score (weighted)
    double combined = (compute_score * 0.4 +
                      memory_score * 0.3 +
                      occupancy_score * 0.3) * utilization_penalty;

    return {compute_score, memory_score, occupancy_score, combined};
}

// Select best tile configuration for given problem size
// Heuristic defaults by problem size:
// - Small problems (M,N < 1024): 32x32 or 64x64 tiles work best
// - Medium problems (1024 <= M,N < 4096): 64x64 tiles with K=32 or 64
// - Large problems (M,N >= 4096): 128x128 with K=64 and 32x32 WMMA
inline const tile::TileParams& select_tile_config(int M, int N, int K) {
    int min_dim = std::min({M, N, K});
    int max_dim = std::max({M, N});

    // For very small problems, use smallest tiles
    if (max_dim <= 512) {
        return tile::TILE_CONFIGS[0];  // Small_32x32x32
    }

    // For small-medium problems
    if (max_dim <= 1024) {
        if (M <= 256) {
            return tile::TILE_CONFIGS[0];  // Small_32x32x32
        }
        return tile::TILE_CONFIGS[3];  // Medium_64x64x32
    }

    // For medium problems
    if (max_dim <= 2048) {
        return tile::TILE_CONFIGS[3];  // Medium_64x64x32
    }

    // For large problems
    if (max_dim <= 4096) {
        return tile::TILE_CONFIGS[8];  // Large_128x128x64
    }

    // For very large problems
    return tile::TILE_CONFIGS[8];  // Large_128x128x64
}

// Autotuning result structure
struct TuningResult {
    int config_index;
    const char* config_name;
    double achieved_tflops;
    double relative_score;
    int bM, bN, bK;
    int threads;

    bool operator<(const TuningResult& other) const {
        return achieved_tflops > other.achieved_tflops;  // Higher is better
    }
};

// Calculate achieved TFLOPS from timing
inline double calc_tflops(int M, int N, int K, double time_ms) {
    double ops = 2.0 * M * N * K;  // GEMM FLOPs
    return ops / (time_ms * 1e-3) / 1e12;  // TFLOPS
}

// Problem size categories for autotuning
enum class ProblemCategory {
    TINY,       // M,N,K < 256
    SMALL,      // 256 <= M,N,K < 1024
    MEDIUM,     // 1024 <= M,N,K < 4096
    LARGE,      // 4096 <= M,N,K < 16384
    XLARGE      // M,N,K >= 16384
};

inline ProblemCategory categorize_problem(int M, int N, int K) {
    int max_dim = std::max({M, N, K});

    if (max_dim < 256) return ProblemCategory::TINY;
    if (max_dim < 1024) return ProblemCategory::SMALL;
    if (max_dim < 4096) return ProblemCategory::MEDIUM;
    if (max_dim < 16384) return ProblemCategory::LARGE;
    return ProblemCategory::XLARGE;
}

// Get recommended configurations for a problem category
inline void get_recommended_configs(ProblemCategory cat,
                                    const tile::TileParams** configs,
                                    int* num_configs) {
    static const tile::TileParams* tiny_configs[] = {
        &tile::TILE_CONFIGS[0],  // Small_32x32x32
        &tile::TILE_CONFIGS[1],  // Small_32x64x32
        &tile::TILE_CONFIGS[2],  // Small_64x32x32
    };

    static const tile::TileParams* small_configs[] = {
        &tile::TILE_CONFIGS[3],  // Medium_64x64x32
        &tile::TILE_CONFIGS[4],  // Medium_64x64x64
        &tile::TILE_CONFIGS[5],  // Medium_64x128x32
        &tile::TILE_CONFIGS[6],  // Medium_128x64x32
    };

    static const tile::TileParams* medium_configs[] = {
        &tile::TILE_CONFIGS[7],  // Large_128x128x32
        &tile::TILE_CONFIGS[8],  // Large_128x128x64
        &tile::TILE_CONFIGS[9],  // Large_128x256x32
        &tile::TILE_CONFIGS[10], // Large_256x128x32
    };

    static const tile::TileParams* large_configs[] = {
        &tile::TILE_CONFIGS[7],  // Large_128x128x32
        &tile::TILE_CONFIGS[11], // XLarge_256x256x32
    };

    switch (cat) {
        case ProblemCategory::TINY:
            *configs = tiny_configs[0];
            *num_configs = 3;
            break;
        case ProblemCategory::SMALL:
            *configs = small_configs[0];
            *num_configs = 4;
            break;
        case ProblemCategory::MEDIUM:
            *configs = medium_configs[0];
            *num_configs = 4;
            break;
        case ProblemCategory::LARGE:
        case ProblemCategory::XLARGE:
            *configs = large_configs[0];
            *num_configs = 2;
            break;
    }
}

} // namespace autotune

// ============================================================================
// Section 8: Combined Tuning Configuration
// ============================================================================

// Complete tuning configuration for a GEMM problem
struct GemmTuningConfig {
    // Tile configuration
    int bM, bN, bK;
    int wmma_m, wmma_n, wmma_k;

    // Execution configuration
    int threads;
    int pipe_stages;

    // Memory configuration
    int vector_width;
    bool use_async_copy;

    // Performance hints
    bool is_memory_bound;
    double estimated_score;
    const char* description;
};

// Get complete tuning configuration for a problem
inline GemmTuningConfig get_tuning_config(int M, int N, int K,
                                          bool is_fp16 = true) {
    // Select tile configuration
    const auto& tile_cfg = autotune::select_tile_config(M, N, K);

    // Select pipeline configuration
    const auto& pipe_cfg = pipeline::select_pipeline(K, tile_cfg.bK);

    // Check if memory-bound
    int elem_bytes = is_fp16 ? 2 : 4;
    bool mem_bound = memory::is_memory_bound(M, N, K, elem_bytes);

    // Estimate relative score
    auto perf_est = autotune::estimate_performance(M, N, K, tile_cfg, is_fp16);

    // Memory configuration
    int vec_width = is_fp16 ? memory::VECTOR_WIDTH_FP16 : memory::VECTOR_WIDTH_FP32;

    return {
        .bM = tile_cfg.bM,
        .bN = tile_cfg.bN,
        .bK = tile_cfg.bK,
        .wmma_m = tile_cfg.wmma_m,
        .wmma_n = tile_cfg.wmma_n,
        .wmma_k = tile_cfg.wmma_k,
        .threads = tile_cfg.threads,
        .pipe_stages = pipe_cfg.stages,
        .vector_width = vec_width,
        .use_async_copy = pipe_cfg.async_copy,
        .is_memory_bound = mem_bound,
        .estimated_score = perf_est.combined_score,
        .description = tile_cfg.name
    };
}

// Print tuning configuration (for debugging)
inline void print_tuning_config(const GemmTuningConfig& cfg, int M, int N, int K) {
    printf("GEMM Tuning Configuration for %dx%dx%d:\n", M, N, K);
    printf("  Tile: %dx%dx%d\n", cfg.bM, cfg.bN, cfg.bK);
    printf("  WMMA: %dx%dx%d\n", cfg.wmma_m, cfg.wmma_n, cfg.wmma_k);
    printf("  Threads: %d, Pipe stages: %d\n", cfg.threads, cfg.pipe_stages);
    printf("  Vector width: %d, Async copy: %s\n",
           cfg.vector_width, cfg.use_async_copy ? "yes" : "no");
    printf("  Memory bound: %s\n", cfg.is_memory_bound ? "yes" : "no");
    printf("  Estimated score: %.3f\n", cfg.estimated_score);
    printf("  Configuration: %s\n", cfg.description);
}

} // namespace flashmoe::tuning

#endif // FLASHMOE_HIP_TUNING_HPP
