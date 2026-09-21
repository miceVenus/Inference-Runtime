#ifndef BBM_THREAD_POOL_H
#define BBM_THREAD_POOL_H

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

// A header-only thread pool with:
//   * a global queue for submissions from external threads;
//   * one local deque per worker;
//   * work stealing between workers;
//   * graceful and cancelling shutdown;
//   * fire-and-forget tasks with an exception callback;
//   * wait(), pause()/resume(), and basic state inspection.


namespace bbm::concurrency{
    class thread_pool {
        public:
            using task_type = std::function<void()>;

            enum class ShutdownMode {
                drain, // finish tasks already accepted
                cancel  // discard tasks which have not started
            };

            struct Options {
                std::size_t thread_count;
                bool work_stealing;
                std::function<void(std::exception_ptr)> unhandled_exception;

                explicit Options(std::size_t threads = default_thread_count())
                    : thread_count(threads),
                    work_stealing(true),
                    unhandled_exception(){}
            };

            static std::size_t default_thread_count() noexcept
            {
                const std::size_t count = std::thread::hardware_concurrency();
                return count == 0 ? 1 : count;
            }

            explicit thread_pool(std::size_t threads = default_thread_count())
                : thread_pool(Options(threads))
            {
            }

            explicit thread_pool(Options options)
                : options_(std::move(options)),
                accepting_(true),
                paused_(false),
                shutdown_requested_(false),
                shutdown_mode_(ShutdownMode::drain),
                queued_tasks_(0),
                outstanding_tasks_(0)
            {
                if (options_.thread_count == 0) {
                    throw std::invalid_argument(
                        "ThreadPool requires at least one worker");
                }

                workers_.reserve(options_.thread_count);
                worker_queues_.reserve(options_.thread_count);

                for (std::size_t i = 0; i < options_.thread_count; ++i) {
                    worker_queues_.emplace_back(new Worker);
                }

                // If thread creation fails, join the workers which were already
                // started. Otherwise their joinable destructors would terminate the
                // process while unwinding this constructor.
                try {
                    for (std::size_t i = 0; i < options_.thread_count; ++i) {
                        workers_.emplace_back([this, i] { worker_loop(i); });
                    }
                } catch (...) {
                    {
                        std::lock_guard<std::mutex> lock(state_mutex_);
                        accepting_ = false;
                        shutdown_requested_ = true;
                    }
                    work_condition_.notify_all();
                    join_workers();
                    throw;
                }
            }

            thread_pool(const thread_pool&) = delete;
            thread_pool& operator=(const thread_pool&) = delete;
            thread_pool(thread_pool&&) = delete;
            thread_pool& operator=(thread_pool&&) = delete;

            ~thread_pool()
            {
                // Destruction is graceful by default: accepted work is not silently
                // abandoned. Do not destroy a pool from one of its own worker tasks;
                // joining the current thread is inherently invalid.
                shutdown(ShutdownMode::drain);
            }

            // Submit a task and return a future for its result. Exceptions thrown by
            // the callable are stored in the future and rethrown by future::get().
            template<class F, class... Args>
            auto enqueue(F&& f, Args&&... args)
                -> std::future<typename std::invoke_result<
                    typename std::decay<F>::type,
                    typename std::decay<Args>::type...>::type>
            {
                using return_type = typename std::invoke_result<
                    typename std::decay<F>::type,
                    typename std::decay<Args>::type...>::type;

                auto bound = std::bind(
                    std::forward<F>(f), std::forward<Args>(args)...);
                auto packaged = std::make_shared<std::packaged_task<return_type()>>(
                    std::move(bound));

                std::future<return_type> result = packaged->get_future();
                submit_task([packaged] { (*packaged)(); });
                return result;
            }

            // Submit a task whose result is intentionally ignored. Any exception
            // escaping the callable is passed to Options::unhandled_exception.
            template<class F, class... Args>
            void enqueue_detached(F&& f, Args&&... args)
            {
                auto bound = std::bind(
                    std::forward<F>(f), std::forward<Args>(args)...);
                auto handler = options_.unhandled_exception;

                submit_task([bound = std::move(bound), handler]() mutable {
                    try {
                        bound();
                    } catch (...) {
                        if (handler) {
                            try {
                                handler(std::current_exception());
                            } catch (...) {
                                // An exception handler must never kill a worker.
                            }
                        }
                    }
                });
            }

            // Alias commonly used by thread-pool APIs.
            template<class F, class... Args>
            void post(F&& f, Args&&... args)
            {
                enqueue_detached(std::forward<F>(f), std::forward<Args>(args)...);
            }

            // Wait until all accepted tasks have either completed or been cancelled.
            // Calling wait() from a worker task is rejected because waiting for the
            // current task would otherwise deadlock the pool.
            void wait()
            {
                if (is_worker_thread()) {
                    throw std::logic_error(
                        "ThreadPool::wait() cannot be called by a worker task");
                }

                std::unique_lock<std::mutex> lock(idle_mutex_);
                idle_condition_.wait(lock, [this] {
                    return outstanding_tasks_.load(std::memory_order_acquire) == 0;
                });
            }

            // Stop accepting work and join all workers. The first shutdown call wins;
            // later calls are harmless and use the already selected mode.
            void shutdown(ShutdownMode mode = ShutdownMode::drain)
            {
                bool first_shutdown = false;
                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    if (!shutdown_requested_) {
                        shutdown_requested_ = true;
                        accepting_ = false;
                        shutdown_mode_ = mode;
                        first_shutdown = true;
                    }
                }

                if (first_shutdown && mode == ShutdownMode::cancel) {
                    cancel_queued_tasks();
                }

                work_condition_.notify_all();
                idle_condition_.notify_all();
                join_workers();
            }

            // Pause takes effect between tasks. It does not interrupt a running task.
            // Shutdown always overrides pause so workers can exit.
            void pause()
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                if (accepting_) {
                    paused_ = true;
                }
            }

            void resume()
            {
                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    paused_ = false;
                }
                work_condition_.notify_all();
            }

            bool accepting() const
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                return accepting_;
            }

            bool paused() const
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                return paused_;
            }

            bool is_worker_thread() const noexcept
            {
                return tls_pool() == this;
            }

            std::size_t thread_count() const noexcept
            {
                return workers_.size();
            }

            // queued_tasks() excludes tasks currently executing.
            std::size_t queued_tasks() const noexcept
            {
                return queued_tasks_.load(std::memory_order_acquire);
            }

            // outstanding_tasks() includes queued and currently executing tasks.
            std::size_t outstanding_tasks() const noexcept
            {
                return outstanding_tasks_.load(std::memory_order_acquire);
            }

        private:
            struct Worker {
                std::mutex mutex;
                std::deque<task_type> tasks;
            };

            static constexpr std::size_t no_worker =
                static_cast<std::size_t>(-1);

            static thread_pool*& tls_pool() noexcept
            {
                static thread_local thread_pool* pool = nullptr;
                return pool;
            }

            static std::size_t& tls_worker_index() noexcept
            {
                static thread_local std::size_t index = no_worker;
                return index;
            }

            bool is_current_worker() const noexcept
            {
                return tls_pool() == this &&
                    tls_worker_index() < worker_queues_.size();
            }

            void submit_task(task_type task)
            {
                if (!task) {
                    throw std::invalid_argument(
                        "ThreadPool cannot enqueue an empty task");
                }

                {
                    // Serializing the acceptance check with queue insertion prevents
                    // shutdown from racing with a task which has just been accepted.
                    std::lock_guard<std::mutex> state_lock(state_mutex_);
                    if (!accepting_) {
                        throw std::runtime_error(
                            "enqueue on stopped ThreadPool");
                    }

                    queued_tasks_.fetch_add(1, std::memory_order_release);
                    outstanding_tasks_.fetch_add(1, std::memory_order_release);

                    try {
                        if (options_.work_stealing && is_current_worker()) {
                            Worker& worker = *worker_queues_[tls_worker_index()];
                            std::lock_guard<std::mutex> queue_lock(worker.mutex);
                            worker.tasks.emplace_back(std::move(task));
                        } else {
                            std::lock_guard<std::mutex> queue_lock(global_mutex_);
                            global_tasks_.emplace_back(std::move(task));
                        }
                    } catch (...) {
                        queued_tasks_.fetch_sub(1, std::memory_order_release);
                        finish_tasks(1);
                        throw;
                    }
                }

                work_condition_.notify_one();
            }

            bool try_pop_local(std::size_t index, task_type& task)
            {
                Worker& worker = *worker_queues_[index];
                std::lock_guard<std::mutex> lock(worker.mutex);
                if (worker.tasks.empty()) {
                    return false;
                }

                // The owner takes the newest task first, which is usually better for
                // cache locality and fork/join-style workloads.
                task = std::move(worker.tasks.back());
                worker.tasks.pop_back();
                return true;
            }

            bool try_pop_global(task_type& task)
            {
                std::lock_guard<std::mutex> lock(global_mutex_);
                if (global_tasks_.empty()) {
                    return false;
                }

                task = std::move(global_tasks_.front());
                global_tasks_.pop_front();
                return true;
            }

            bool try_steal(std::size_t thief,
                        std::size_t victim,
                        task_type& task)
            {
                if (thief == victim) {
                    return false;
                }

                Worker& worker = *worker_queues_[victim];
                std::lock_guard<std::mutex> lock(worker.mutex);
                if (worker.tasks.empty()) {
                    return false;
                }

                // Thieves take the oldest task, reducing interference with the owner.
                task = std::move(worker.tasks.front());
                worker.tasks.pop_front();
                return true;
            }

            bool try_get_task(std::size_t index,
                            std::minstd_rand& random,
                            task_type& task)
            {
                if (options_.work_stealing && try_pop_local(index, task)) {
                    return true;
                }

                if (try_pop_global(task)) {
                    return true;
                }

                if (!options_.work_stealing || worker_queues_.size() <= 1) {
                    return false;
                }

                const std::size_t count = worker_queues_.size();
                const std::size_t first_victim =
                    static_cast<std::size_t>(random()) % count;

                for (std::size_t offset = 0; offset < count; ++offset) {
                    const std::size_t victim = (first_victim + offset) % count;
                    if (try_steal(index, victim, task)) {
                        return true;
                    }
                }

                return false;
            }

            void worker_loop(std::size_t index) noexcept
            {
                // Track this worker so nested submissions can use its local queue.
                tls_pool() = this;
                tls_worker_index() = index;

                std::minstd_rand random(
                    static_cast<unsigned>(index + 1) * 2654435761u);

                for (;;) {
                    {
                        std::unique_lock<std::mutex> lock(state_mutex_);
                        work_condition_.wait(lock, [this] {
                            return !accepting_ ||
                                (!paused_ &&
                                    queued_tasks_.load(std::memory_order_acquire) != 0);
                        });

                        if (!accepting_ &&
                            queued_tasks_.load(std::memory_order_acquire) == 0) {
                            break;
                        }
                    }

                    task_type task;
                    if (!try_get_task(index, random, task)) {
                        // Another worker may have taken the last task after the
                        // condition-variable predicate was checked.
                        continue;
                    }

                    // The task is now running, so it is no longer queued.
                    queued_tasks_.fetch_sub(1, std::memory_order_release);

                    try {
                        task();
                    } catch (...) {
                        report_unhandled_exception(std::current_exception());
                    }

                    finish_tasks(1);
                }

                tls_worker_index() = no_worker;
                tls_pool() = nullptr;
            }

            void report_unhandled_exception(std::exception_ptr exception) noexcept
            {
                if (!options_.unhandled_exception) {
                    return;
                }

                try {
                    options_.unhandled_exception(exception);
                } catch (...) {
                    // User callbacks must not terminate the worker loop.
                }
            }

            void finish_tasks(std::size_t count) noexcept
            {
                std::size_t previous;
                {
                    // Pair with wait() so the last completion cannot be missed.
                    std::lock_guard<std::mutex> lock(idle_mutex_);
                    previous = outstanding_tasks_.fetch_sub(
                        count, std::memory_order_acq_rel);
                }
                if (previous == count) {
                    idle_condition_.notify_all();
                }
            }

            void cancel_queued_tasks()
            {
                std::vector<task_type> discarded;

                {
                    std::lock_guard<std::mutex> lock(global_mutex_);
                    while (!global_tasks_.empty()) {
                        discarded.emplace_back(std::move(global_tasks_.front()));
                        global_tasks_.pop_front();
                    }
                }

                for (const std::unique_ptr<Worker>& worker : worker_queues_) {
                    std::lock_guard<std::mutex> lock(worker->mutex);
                    while (!worker->tasks.empty()) {
                        discarded.emplace_back(std::move(worker->tasks.front()));
                        worker->tasks.pop_front();
                    }
                }

                if (discarded.empty()) {
                    return;
                }

                const std::size_t count = discarded.size();
                // Destroy packaged_task wrappers before publishing the counters. This
                // makes their futures report broken_promise before wait() can return.
                discarded.clear();
                queued_tasks_.fetch_sub(count, std::memory_order_release);
                finish_tasks(count);
            }

            void join_workers()
            {
                std::lock_guard<std::mutex> lock(join_mutex_);
                for (std::thread& worker : workers_) {
                    if (worker.joinable()) {
                        worker.join();
                    }
                }
            }

        private:
            Options options_;

            std::vector<std::thread> workers_;
            std::vector<std::unique_ptr<Worker>> worker_queues_;

            std::mutex global_mutex_;
            std::deque<task_type> global_tasks_;

            mutable std::mutex state_mutex_;
            bool accepting_;
            bool paused_;
            bool shutdown_requested_;
            ShutdownMode shutdown_mode_;

            std::condition_variable work_condition_;
            std::atomic<std::size_t> queued_tasks_;

            std::mutex idle_mutex_;
            std::condition_variable idle_condition_;
            std::atomic<std::size_t> outstanding_tasks_;

            std::mutex join_mutex_;
};
}

#endif