/*
 * Stub <omp.h> for building the upstream GENESIS C++ core WITHOUT OpenMP.
 *
 * Upstream's EM_EM.cpp unconditionally does `#include <omp.h>` and calls
 * omp_set_num_threads().  When no OpenMP runtime is available, put this
 * directory on the include path: the `#pragma omp ...` lines are then ignored
 * by the compiler and every loop runs serially.  Numerical results are the
 * serial ones (identical to an OpenMP build run with num_threads = 1).
 */
#ifndef GENESIS_ORACLE_OMP_STUB_H
#define GENESIS_ORACLE_OMP_STUB_H

static inline void omp_set_num_threads(int) {}
static inline int omp_get_max_threads(void) { return 1; }
static inline int omp_get_num_threads(void) { return 1; }
static inline int omp_get_thread_num(void) { return 0; }

#endif /* GENESIS_ORACLE_OMP_STUB_H */
