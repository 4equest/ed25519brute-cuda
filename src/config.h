#pragma once

// ============================================================
// CUDA Kernel Configuration
// ============================================================
// These parameters control kernel launch configuration and workload distribution.
// All values are compile-time constants for optimal optimization.

// --- Kernel Launch Configuration ---
#define THREADS_PER_BLOCK      256    // Threads per block (power of 2, typically 128-512)
#define MIN_BLOCKS_PER_SM      2      // Min blocks per SM for __launch_bounds__ occupancy hint

// --- Kernel Workload Configuration ---
#define ITERATIONS_PER_THREAD  256    // Keys checked per thread per kernel launch
#define BATCH_SIZE             32     // Batch inversion size (Montgomery's trick)

// --- Runtime Configuration Defaults ---
#define DEFAULT_BLOCKS         256    // Default grid blocks (can be overridden via --blocks)
#define MAX_BLOCKS             4096   // Maximum allowed blocks

// --- Host-side Configuration ---
#define NUM_STREAMS            2      // CUDA streams for double buffering
#define PROGRESS_REPORT_INTERVAL 50000000ULL  // Progress report every N keys

// ============================================================
// Performance Notes
// ============================================================
// - THREADS_PER_BLOCK: 256 is optimal for most GPUs (good occupancy)
// - ITERATIONS_PER_THREAD: Higher = less kernel launch overhead, but less responsive
// - BATCH_SIZE: 32 provides good balance for Montgomery batch inversion
// - NUM_STREAMS: 2 enables overlapping kernel execution with result checking
