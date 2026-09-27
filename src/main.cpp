#include "cpu_parallel.hpp"
#include "executor.hpp"
#include "loader.hpp"
#include "tensor.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t image_rows = 28;
constexpr std::size_t image_columns = 28;
constexpr std::size_t pixels_per_image = image_rows * image_columns;
constexpr std::size_t warmup_count = 5;

struct MnistDataset {
    std::vector<std::uint8_t> images;
    std::vector<std::uint8_t> labels;
};

std::uint32_t read_big_endian_u32(std::istream& input, const char* field_name) {
    std::array<std::uint8_t, 4> bytes{};
    input.read(reinterpret_cast<char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
    if (input.gcount() != static_cast<std::streamsize>(bytes.size())) {
        throw std::runtime_error(std::string("truncated MNIST IDX header: ") + field_name);
    }

    return (static_cast<std::uint32_t>(bytes[0]) << 24) |
        (static_cast<std::uint32_t>(bytes[1]) << 16) |
        (static_cast<std::uint32_t>(bytes[2]) << 8) |
        static_cast<std::uint32_t>(bytes[3]);
}

MnistDataset load_mnist_dataset(
    std::filesystem::path directory,
    const std::string& split) {
    const bool training = split == "train";
    const std::string image_filename = training
        ? "train-images-idx3-ubyte"
        : "t10k-images-idx3-ubyte";
    const std::string label_filename = training
        ? "train-labels-idx1-ubyte"
        : "t10k-labels-idx1-ubyte";

    // Accept either the MNIST directory or its raw/ subdirectory.
    if (!std::filesystem::exists(directory / image_filename) &&
        std::filesystem::exists(directory / "raw" / image_filename)) {
        directory /= "raw";
    }

    const auto image_path = directory / image_filename;
    const auto label_path = directory / label_filename;
    std::ifstream image_file(image_path, std::ios::binary);
    if (!image_file) {
        throw std::runtime_error("cannot open MNIST image file: " + image_path.string());
    }
    std::ifstream label_file(label_path, std::ios::binary);
    if (!label_file) {
        throw std::runtime_error("cannot open MNIST label file: " + label_path.string());
    }

    if (read_big_endian_u32(image_file, "image magic") != 2051) {
        throw std::runtime_error("invalid MNIST image IDX magic: " + image_path.string());
    }
    const std::uint32_t image_count = read_big_endian_u32(image_file, "image count");
    const std::uint32_t rows = read_big_endian_u32(image_file, "image rows");
    const std::uint32_t columns = read_big_endian_u32(image_file, "image columns");
    if (rows != image_rows || columns != image_columns) {
        throw std::runtime_error("MNIST images must be 28x28: " + image_path.string());
    }
    if (image_count == 0 ||
        image_count > std::numeric_limits<std::size_t>::max() / pixels_per_image) {
        throw std::runtime_error("invalid MNIST image count: " + image_path.string());
    }

    if (read_big_endian_u32(label_file, "label magic") != 2049) {
        throw std::runtime_error("invalid MNIST label IDX magic: " + label_path.string());
    }
    const std::uint32_t label_count = read_big_endian_u32(label_file, "label count");
    if (label_count != image_count) {
        throw std::runtime_error("MNIST image and label counts do not match");
    }

    MnistDataset dataset;
    dataset.images.resize(static_cast<std::size_t>(image_count) * pixels_per_image);
    dataset.labels.resize(label_count);

    const auto image_bytes = static_cast<std::streamsize>(dataset.images.size());
    image_file.read(reinterpret_cast<char*>(dataset.images.data()), image_bytes);
    if (image_file.gcount() != image_bytes) {
        throw std::runtime_error("truncated MNIST image data: " + image_path.string());
    }

    const auto label_bytes = static_cast<std::streamsize>(dataset.labels.size());
    label_file.read(reinterpret_cast<char*>(dataset.labels.data()), label_bytes);
    if (label_file.gcount() != label_bytes) {
        throw std::runtime_error("truncated MNIST label data: " + label_path.string());
    }
    if (std::any_of(dataset.labels.begin(), dataset.labels.end(),
            [](std::uint8_t label) { return label > 9; })) {
        throw std::runtime_error("MNIST labels must be in the range 0..9");
    }

    return dataset;
}

std::size_t parse_positive_size(const char* text, const char* name) {
    const std::string value(text);
    if (value.empty() || value.front() == '-') {
        throw std::invalid_argument(std::string(name) + " must be a positive integer");
    }

    std::size_t parsed_characters = 0;
    const unsigned long long parsed = std::stoull(value, &parsed_characters);
    if (parsed_characters != value.size() || parsed == 0 ||
        parsed > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(std::string(name) + " must be a positive integer");
    }
    return static_cast<std::size_t>(parsed);
}

Tensor make_batch(
    const MnistDataset& dataset,
    std::size_t first_image,
    std::size_t valid_count,
    std::size_t batch_size) {
    std::vector<Float32> values(batch_size * pixels_per_image);
    for (std::size_t item = 0; item < batch_size; ++item) {
        // A fixed-shape graph needs a full final batch. Pad with its last valid image.
        const std::size_t image_index = first_image + std::min(item, valid_count - 1);
        const std::size_t source_offset = image_index * pixels_per_image;
        const std::size_t destination_offset = item * pixels_per_image;
        for (std::size_t pixel = 0; pixel < pixels_per_image; ++pixel) {
            const Float32 normalized =
                static_cast<Float32>(dataset.images[source_offset + pixel]) / 255.0f;
            values[destination_offset + pixel] = (normalized - 0.1307f) / 0.3081f;
        }
    }

    return Tensor(
        Shape({static_cast<long>(batch_size), static_cast<long>(pixels_per_image)}),
        std::move(values),
        Dtype::Float32);
}

double percentile(const std::vector<double>& sorted_values, double p) {
    const auto index = static_cast<std::size_t>(
        std::ceil(p * static_cast<double>(sorted_values.size())) - 1.0);
    return sorted_values[std::min(index, sorted_values.size() - 1)];
}

double median(const std::vector<double>& sorted_values) {
    const std::size_t middle = sorted_values.size() / 2;
    if (sorted_values.size() % 2 != 0) {
        return sorted_values[middle];
    }
    return (sorted_values[middle - 1] + sorted_values[middle]) / 2.0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 6) {
        std::cerr << "usage: runtime_demo MODEL.onnx MNIST_DIR [train|test] "
                     "[BATCH_SIZE] [MAX_SAMPLES]\n";
        return 2;
    }

    try {
        const std::filesystem::path model_path(argv[1]);
        const std::filesystem::path dataset_directory(argv[2]);
        const std::string split = argc >= 4 ? argv[3] : "test";
        if (split != "train" && split != "test") {
            throw std::invalid_argument("split must be 'train' or 'test'");
        }

        constexpr std::size_t default_batch_size = 16;
        const std::size_t batch_size = argc >= 5
            ? parse_positive_size(argv[4], "batch size")
            : default_batch_size;
        if (batch_size > static_cast<std::size_t>(std::numeric_limits<long>::max()) ||
            batch_size > std::numeric_limits<std::size_t>::max() / pixels_per_image) {
            throw std::invalid_argument("batch size is too large");
        }

        const MnistDataset dataset = load_mnist_dataset(dataset_directory, split);
        const std::size_t requested_samples = argc == 6
            ? parse_positive_size(argv[5], "sample count")
            : dataset.labels.size();
        const std::size_t sample_count =
            std::min(requested_samples, dataset.labels.size());
        const std::size_t batch_count = sample_count / batch_size +
            (sample_count % batch_size == 0 ? 0 : 1);

        Graph graph = Loader::load(
            model_path.string(), static_cast<long>(batch_size));
        if (graph.inputs().size() != 1 || graph.outputs().size() != 1) {
            throw std::runtime_error("benchmark expects one input and one output");
        }

        const auto input_id = graph.inputs().front();
        const auto output_id = graph.outputs().front();
        const auto output_name = graph.value(output_id).name();
        Executor executor(graph);
        executor.set_input(input_id,
            make_batch(dataset, 0, std::min(batch_size, sample_count), batch_size));

        auto& pool = cpu_runtime::worker_pool();
        // Warm the worker pool and model kernels before collecting measurements.
        for (std::size_t i = 0; i < warmup_count; ++i) {
            executor.run();
        }

        std::vector<double> batch_latencies_ms;
        batch_latencies_ms.reserve(batch_count);
        double total_run_seconds = 0.0;
        std::size_t correct = 0;

        for (std::size_t first_image = 0;
             first_image < sample_count;
             first_image += batch_size) {
            const std::size_t valid_count =
                std::min(batch_size, sample_count - first_image);
            executor.set_input(input_id,
                make_batch(dataset, first_image, valid_count, batch_size));

            const auto start = std::chrono::steady_clock::now();
            executor.run();
            const auto stop = std::chrono::steady_clock::now();
            const double elapsed_seconds =
                std::chrono::duration<double>(stop - start).count();
            total_run_seconds += elapsed_seconds;
            batch_latencies_ms.push_back(elapsed_seconds * 1000.0);

            const Tensor& output = executor.get_output(output_name);
            if (output.numel() <= 0 ||
                static_cast<std::size_t>(output.numel()) % batch_size != 0) {
                throw std::runtime_error(
                    "model output must contain an equal, non-empty class row per image");
            }
            const std::size_t classes_per_image =
                static_cast<std::size_t>(output.numel()) / batch_size;
            for (std::size_t item = 0; item < valid_count; ++item) {
                const Float32* row = output.raw_data() + item * classes_per_image;
                const auto best = std::max_element(row, row + classes_per_image);
                const std::size_t prediction = static_cast<std::size_t>(best - row);
                if (prediction == dataset.labels[first_image + item]) {
                    ++correct;
                }
            }
        }

        std::sort(batch_latencies_ms.begin(), batch_latencies_ms.end());
        const double total_run_ms = std::accumulate(
            batch_latencies_ms.begin(), batch_latencies_ms.end(), 0.0);
        const double mean_batch_ms = total_run_ms / batch_latencies_ms.size();
        const double accuracy = 100.0 * static_cast<double>(correct) / sample_count;

        std::cout << std::fixed << std::setprecision(3)
                  << "model: " << model_path << '\n'
                  << "MNIST split: " << split << '\n'
                  << "images measured: " << sample_count << '\n'
                  << "batch size: " << batch_size << '\n'
                  << "measured batches: " << batch_count << '\n'
                  << "warmups: " << warmup_count << '\n'
                  << "CPU workers: " << pool.thread_count() << '\n'
                  << "AVX2+FMA: "
                  << (cpu_runtime::avx2_fma_available() ? "yes" : "no") << '\n'
                  << "run latency ms/batch: mean=" << mean_batch_ms
                  << ", median=" << median(batch_latencies_ms)
                  << ", p95=" << percentile(batch_latencies_ms, 0.95)
                  << ", min=" << batch_latencies_ms.front()
                  << ", max=" << batch_latencies_ms.back() << '\n'
                  << "throughput images/s (Executor::run only): "
                  << (static_cast<double>(sample_count) / total_run_seconds) << '\n'
                  << "accuracy: " << correct << '/' << sample_count
                  << " (" << accuracy << "%)\n";
        if (sample_count % batch_size != 0) {
            std::cout << "note: the final fixed-size batch was padded with its last "
                         "valid image; accuracy counts only dataset images\n";
        }
    } catch (const std::exception& error) {
        std::cerr << "benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
