/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Flashmoe Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 *
 * NVSHMEM-to-ROCSHMEM Compatibility Layer
 * Provides mappings from NVSHMEM primitives to ROCSHMEM equivalents for AMD GPU support.
 */

#ifndef FLASHMOE_HIP_NVSHMEM_COMPAT_HPP
#define FLASHMOE_HIP_NVSHMEM_COMPAT_HPP

#ifdef FLASHMOE_HAS_ROCSHMEM
#include <rocshmem/rocshmem.hpp>

// Import rocshmem namespace for convenience
using namespace rocshmem;

// ============================================================================
// Signal Operation Constants
// ============================================================================

#ifndef NVSHMEM_SIGNAL_SET
#define NVSHMEM_SIGNAL_SET ROCSHMEM_SIGNAL_SET
#endif

#ifndef NVSHMEM_SIGNAL_ADD
#define NVSHMEM_SIGNAL_ADD ROCSHMEM_SIGNAL_ADD
#endif

// ============================================================================
// Initialization and Finalization
// ============================================================================

inline int nvshmem_init() {
    rocshmem_init();
    return 0; // NVSHMEM returns void, but some wrappers expect int
}

inline void nvshmem_finalize() {
    rocshmem_finalize();
}

// ============================================================================
// PE (Processing Element) Query Functions
// ============================================================================

inline int nvshmem_my_pe() {
    return rocshmem_my_pe();
}

inline int nvshmem_n_pes() {
    return rocshmem_n_pes();
}

// ============================================================================
// Memory Management
// ============================================================================

inline void* nvshmem_malloc(size_t size) {
    return rocshmem_malloc(size);
}

inline void nvshmem_free(void* ptr) {
    rocshmem_free(ptr);
}

// Aligned allocation
inline void* nvshmem_align(size_t alignment, size_t size) {
    return rocshmem_align(alignment, size);
}

// Callable allocation (device-accessible allocation routine)
inline void* nvshmem_calloc(size_t count, size_t size) {
    return rocshmem_calloc(count, size);
}

// ============================================================================
// Remote Pointer Access
// ============================================================================

inline void* nvshmem_ptr(const void* dest, int pe) {
    return rocshmem_ptr(dest, pe);
}

// ============================================================================
// Memory Synchronization
// ============================================================================

// Fence ensures ordering of remote memory operations
__device__ __forceinline__
void nvshmem_fence() {
    rocshmem_fence();
}

// Quiet ensures all outstanding remote memory operations have completed
__device__ __forceinline__
void nvshmem_quiet() {
    rocshmem_quiet();
}

// Host-callable versions
inline void nvshmem_fence_host() {
    rocshmem_fence();
}

inline void nvshmem_quiet_host() {
    rocshmem_quiet();
}

// ============================================================================
// Barrier Operations
// ============================================================================

__device__ __forceinline__
void nvshmem_barrier_all() {
    rocshmem_barrier_all();
}

inline void nvshmem_barrier_all_host() {
    rocshmem_barrier_all();
}

// ============================================================================
// Put Operations (Non-blocking)
// ============================================================================

template<typename T>
__device__ __forceinline__
void nvshmem_put_nbi(T* dest, const T* source, size_t nelems, int pe) {
    rocshmem_put_nbi(dest, source, nelems, pe);
}

__device__ __forceinline__
void nvshmem_putmem_nbi(void* dest, const void* source, size_t nelems, int pe) {
    rocshmem_putmem_nbi(dest, source, nelems, pe);
}

// ============================================================================
// Put with Signal Operations (Non-blocking Immediate)
// ============================================================================

// The key function for overlapping communication with computation
// Puts data and then signals completion to the target PE

template<typename T>
__device__ __forceinline__
void nvshmem_put_signal_nbi(T* dest, const T* source, size_t nelems,
                             uint64_t* sig_addr, uint64_t signal, int sig_op, int pe) {
    rocshmem_put_signal_nbi(dest, source, nelems, sig_addr, signal, sig_op, pe);
}

__device__ __forceinline__
void nvshmem_putmem_signal_nbi(void* dest, const void* source, size_t nelems,
                                uint64_t* sig_addr, uint64_t signal, int sig_op, int pe) {
    rocshmem_putmem_signal_nbi(dest, source, nelems, sig_addr, signal, sig_op, pe);
}

// ============================================================================
// Workgroup and Wavefront Collective Variants (AMD-specific optimization)
// ============================================================================

// Workgroup-collective put with signal - better for when all threads in workgroup participate
__device__ __forceinline__
void nvshmem_putmem_signal_nbi_wg(void* dest, const void* source, size_t nelems,
                                   uint64_t* sig_addr, uint64_t signal, int sig_op, int pe) {
    rocshmem_putmem_signal_nbi_wg(dest, source, nelems, sig_addr, signal, sig_op, pe);
}

// Wavefront-collective put with signal - better for when only a wavefront participates
__device__ __forceinline__
void nvshmem_putmem_signal_nbi_wave(void* dest, const void* source, size_t nelems,
                                     uint64_t* sig_addr, uint64_t signal, int sig_op, int pe) {
    rocshmem_putmem_signal_nbi_wave(dest, source, nelems, sig_addr, signal, sig_op, pe);
}

// ============================================================================
// Signal-Only Operations (NVSHMEM Extension API)
// ============================================================================

// nvshmemx_signal_op - signal-only operation without data transfer
// Used when you only need to set a signal on a remote PE without transferring data
__device__ __forceinline__
void nvshmemx_signal_op(uint64_t* sig_addr, uint64_t signal, int sig_op, int pe) {
    rocshmem_signal_op(sig_addr, signal, sig_op, pe);
}

// Workgroup-collective version
__device__ __forceinline__
void nvshmemx_signal_op_wg(uint64_t* sig_addr, uint64_t signal, int sig_op, int pe) {
    rocshmem_signal_op_wg(sig_addr, signal, sig_op, pe);
}

// Wavefront-collective version
__device__ __forceinline__
void nvshmemx_signal_op_wave(uint64_t* sig_addr, uint64_t signal, int sig_op, int pe) {
    rocshmem_signal_op_wave(sig_addr, signal, sig_op, pe);
}

// ============================================================================
// Get Operations (Non-blocking)
// ============================================================================

template<typename T>
__device__ __forceinline__
void nvshmem_get_nbi(T* dest, const T* source, size_t nelems, int pe) {
    rocshmem_get_nbi(dest, source, nelems, pe);
}

__device__ __forceinline__
void nvshmem_getmem_nbi(void* dest, const void* source, size_t nelems, int pe) {
    rocshmem_getmem_nbi(dest, source, nelems, pe);
}

// ============================================================================
// Blocking Put/Get Operations
// ============================================================================

template<typename T>
__device__ __forceinline__
void nvshmem_put(T* dest, const T* source, size_t nelems, int pe) {
    rocshmem_put(dest, source, nelems, pe);
}

__device__ __forceinline__
void nvshmem_putmem(void* dest, const void* source, size_t nelems, int pe) {
    rocshmem_putmem(dest, source, nelems, pe);
}

template<typename T>
__device__ __forceinline__
void nvshmem_get(T* dest, const T* source, size_t nelems, int pe) {
    rocshmem_get(dest, source, nelems, pe);
}

__device__ __forceinline__
void nvshmem_getmem(void* dest, const void* source, size_t nelems, int pe) {
    rocshmem_getmem(dest, source, nelems, pe);
}

// ============================================================================
// Atomic Operations
// ============================================================================

template<typename T>
__device__ __forceinline__
T nvshmem_atomic_fetch_add(T* dest, T value, int pe) {
    return rocshmem_atomic_fetch_add(dest, value, pe);
}

template<typename T>
__device__ __forceinline__
T nvshmem_atomic_fetch_inc(T* dest, int pe) {
    return rocshmem_atomic_fetch_inc(dest, pe);
}

template<typename T>
__device__ __forceinline__
T nvshmem_atomic_swap(T* dest, T value, int pe) {
    return rocshmem_atomic_swap(dest, value, pe);
}

template<typename T>
__device__ __forceinline__
T nvshmem_atomic_compare_swap(T* dest, T cond, T value, int pe) {
    return rocshmem_atomic_compare_swap(dest, cond, value, pe);
}

template<typename T>
__device__ __forceinline__
void nvshmem_atomic_set(T* dest, T value, int pe) {
    rocshmem_atomic_set(dest, value, pe);
}

template<typename T>
__device__ __forceinline__
T nvshmem_atomic_fetch(const T* source, int pe) {
    return rocshmem_atomic_fetch(source, pe);
}

// ============================================================================
// Wait/Test Operations
// ============================================================================

template<typename T>
__device__ __forceinline__
void nvshmem_wait_until(T* ivar, int cmp, T cmp_value) {
    rocshmem_wait_until(ivar, cmp, cmp_value);
}

template<typename T>
__device__ __forceinline__
int nvshmem_test(T* ivar, int cmp, T cmp_value) {
    return rocshmem_test(ivar, cmp, cmp_value);
}

// Signal wait - blocks until signal condition is met
__device__ __forceinline__
uint64_t nvshmem_signal_wait_until(uint64_t* sig_addr, int cmp, uint64_t cmp_value) {
    return rocshmem_signal_wait_until(sig_addr, cmp, cmp_value);
}

// Signal fetch - non-blocking read of a signal value
// Returns the current value of the signal without blocking
__device__ __forceinline__
uint64_t nvshmem_signal_fetch(const uint64_t* sig_addr) {
    return rocshmem_signal_fetch(sig_addr);
}

// ============================================================================
// Type-Specific Test Operations
// ============================================================================

// nvshmem_uint64_test - Non-blocking test if signal matches condition
// Returns 1 if condition is met, 0 otherwise
__device__ __forceinline__
int nvshmem_uint64_test(uint64_t* ivar, int cmp, uint64_t cmp_value) {
    return rocshmem_uint64_test(ivar, cmp, cmp_value);
}

// nvshmem_int_test
__device__ __forceinline__
int nvshmem_int_test(int* ivar, int cmp, int cmp_value) {
    return rocshmem_int_test(ivar, cmp, cmp_value);
}

// nvshmem_long_test
__device__ __forceinline__
int nvshmem_long_test(long* ivar, int cmp, long cmp_value) {
    return rocshmem_long_test(ivar, cmp, cmp_value);
}

// ============================================================================
// Comparison Constants for Wait/Test
// ============================================================================

#ifndef NVSHMEM_CMP_EQ
#define NVSHMEM_CMP_EQ  ROCSHMEM_CMP_EQ
#endif

#ifndef NVSHMEM_CMP_NE
#define NVSHMEM_CMP_NE  ROCSHMEM_CMP_NE
#endif

#ifndef NVSHMEM_CMP_GT
#define NVSHMEM_CMP_GT  ROCSHMEM_CMP_GT
#endif

#ifndef NVSHMEM_CMP_GE
#define NVSHMEM_CMP_GE  ROCSHMEM_CMP_GE
#endif

#ifndef NVSHMEM_CMP_LT
#define NVSHMEM_CMP_LT  ROCSHMEM_CMP_LT
#endif

#ifndef NVSHMEM_CMP_LE
#define NVSHMEM_CMP_LE  ROCSHMEM_CMP_LE
#endif

// ============================================================================
// Team Operations
// ============================================================================

// Type alias for team handle
using nvshmem_team_t = rocshmem_team_t;

// Predefined teams
#define NVSHMEM_TEAM_WORLD ROCSHMEM_TEAM_WORLD

inline int nvshmem_team_my_pe(nvshmem_team_t team) {
    return rocshmem_team_my_pe(team);
}

inline int nvshmem_team_n_pes(nvshmem_team_t team) {
    return rocshmem_team_n_pes(team);
}

// ============================================================================
// Collective Operations
// ============================================================================

template<typename T>
__device__ __forceinline__
void nvshmem_broadcast(T* dest, const T* source, size_t nelems, int pe_root, int pe_start,
                        int log_pe_stride, int pe_size, long* pSync) {
    rocshmem_broadcast(dest, source, nelems, pe_root, pe_start, log_pe_stride, pe_size, pSync);
}

template<typename T>
__device__ __forceinline__
void nvshmem_alltoall(T* dest, const T* source, size_t nelems) {
    rocshmem_alltoall(dest, source, nelems);
}

// ============================================================================
// Type-specific Operations (Convenience Wrappers)
// ============================================================================

// 32-bit operations
__device__ __forceinline__
void nvshmem_int_put(int* dest, const int* source, size_t nelems, int pe) {
    rocshmem_int_put(dest, source, nelems, pe);
}

__device__ __forceinline__
void nvshmem_int_get(int* dest, const int* source, size_t nelems, int pe) {
    rocshmem_int_get(dest, source, nelems, pe);
}

__device__ __forceinline__
int nvshmem_int_atomic_fetch_add(int* dest, int value, int pe) {
    return rocshmem_int_atomic_fetch_add(dest, value, pe);
}

// 64-bit operations
__device__ __forceinline__
void nvshmem_long_put(long* dest, const long* source, size_t nelems, int pe) {
    rocshmem_long_put(dest, source, nelems, pe);
}

__device__ __forceinline__
void nvshmem_long_get(long* dest, const long* source, size_t nelems, int pe) {
    rocshmem_long_get(dest, source, nelems, pe);
}

__device__ __forceinline__
long nvshmem_long_atomic_fetch_add(long* dest, long value, int pe) {
    return rocshmem_long_atomic_fetch_add(dest, value, pe);
}

// Float operations
__device__ __forceinline__
void nvshmem_float_put(float* dest, const float* source, size_t nelems, int pe) {
    rocshmem_float_put(dest, source, nelems, pe);
}

__device__ __forceinline__
void nvshmem_float_get(float* dest, const float* source, size_t nelems, int pe) {
    rocshmem_float_get(dest, source, nelems, pe);
}

// Double operations
__device__ __forceinline__
void nvshmem_double_put(double* dest, const double* source, size_t nelems, int pe) {
    rocshmem_double_put(dest, source, nelems, pe);
}

__device__ __forceinline__
void nvshmem_double_get(double* dest, const double* source, size_t nelems, int pe) {
    rocshmem_double_get(dest, source, nelems, pe);
}

#else // !FLASHMOE_HAS_ROCSHMEM

#define NVSHMEM_SIGNAL_SET 0
#define NVSHMEM_CMP_EQ 0

__device__ __forceinline__
void nvshmem_putmem_signal_nbi(void*, const void*, size_t, uint64_t*, uint64_t, int, int) {
    __builtin_trap();
}

__device__ __forceinline__
void nvshmemx_signal_op(uint64_t*, uint64_t, int, int) {
    __builtin_trap();
}

__device__ __forceinline__
uint64_t nvshmem_signal_fetch(const uint64_t* sig) {
    return *sig;
}

__device__ __forceinline__
bool nvshmem_uint64_test(const uint64_t*, int, uint64_t) {
    return true;
}

__device__ __forceinline__
void nvshmem_fence() {}

__device__ __forceinline__
void* nvshmem_ptr(void* ptr, int) { return ptr; }

__device__ __forceinline__
void* rocshmem_ptr(void* ptr, int) { return ptr; }

__host__ __forceinline__
void* rocshmem_calloc(size_t, size_t) { return nullptr; }

__host__ __forceinline__
void* rocshmem_malloc(size_t) { return nullptr; }

__host__ __forceinline__
void rocshmem_free(void*) {}

#endif // FLASHMOE_HAS_ROCSHMEM
#endif // FLASHMOE_HIP_NVSHMEM_COMPAT_HPP
