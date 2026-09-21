#ifndef CPU_PARALLEL_HPP
#define CPU_PARALLEL_HPP

#include "thread_pool.hpp"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <future>
#include <vector>

namespace cpu_runtime {

// Reuse one pool across operators instead of creating threads per call.
inline bbm::concurrency::thread_pool& worker_pool() {
    static bbm::concurrency::thread_pool pool;
    return pool;
}

inline bool avx2_fma_available() noexcept {
#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
    // Cache feature detection before dispatching AVX2 kernels.
    static const bool available = [] {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2") &&
            __builtin_cpu_supports("fma");
    }();
    return available;
#else
    return false;
#endif
}

template <class Function>
void parallel_for(
    std::size_t count,
    std::size_t minimum_grain,
    Function&& function) {
    if (count == 0) {
        return;
    }

    minimum_grain = std::max<std::size_t>(minimum_grain, 1);
    // Keep small spans serial to avoid task scheduling overhead.
    if (count <= minimum_grain) {
        function(0, count);
        return;
    }

    auto& pool = worker_pool();
    // Run nested work inline to avoid workers waiting on their own pool.
    if (pool.is_worker_thread() || pool.thread_count() < 2) {
        function(0, count);
        return;
    }

    const std::size_t desired_tasks = 1 + (count - 1) / minimum_grain;
    const std::size_t task_count =
        std::min(pool.thread_count(), desired_tasks);
    if (task_count < 2) {
        function(0, count);
        return;
    }

    // Distribute the remainder across the first chunks for balanced work.
    const std::size_t base_chunk_size = count / task_count;
    const std::size_t larger_chunks = count % task_count;
    std::vector<std::future<void>> futures;
    futures.reserve(task_count);

    std::exception_ptr failure;
    std::size_t begin = 0;
    for (std::size_t task = 0; task < task_count; ++task) {
        const std::size_t chunk_size =
            base_chunk_size + (task < larger_chunks ? 1 : 0);
        const std::size_t end = begin + chunk_size;
        try {
            futures.emplace_back(pool.enqueue([&function, begin, end] {
                function(begin, end);
            }));
        } catch (...) {
            failure = std::current_exception();
            break;
        }
        begin = end;
    }

    // Join every task and propagate worker failures to the caller.
    for (auto& future : futures) {
        try {
            future.get();
        } catch (...) {
            if (!failure) {
                failure = std::current_exception();
            }
        }
    }

    if (failure) {
        std::rethrow_exception(failure);
    }
}

}  // namespace cpu_runtime

#endif
