#include "executor.hpp"
#include "loader.hpp"
#include "tensor.hpp"

#include <png.h>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <ctime>

namespace {

std::filesystem::path find_file(const std::filesystem::path& path) {
    if (std::filesystem::exists(path)) {
        return path;
    }

    const auto from_build = std::filesystem::path("..") / path;
    if (std::filesystem::exists(from_build)) {
        return from_build;
    }

    throw std::runtime_error("file not found: " + path.string());
}

}  // namespace

int main() {
    constexpr const char* model_name =
        "experiments/mnist_onnx/artifacts/mnist_heavy_mlp.onnx";
    constexpr const char* image_name = "test_samples/mnist_test_00000_label_7.png";

    try {
        const auto model_path = find_file(model_name);
        const auto image_path = find_file(image_name);

        Tensor image = load_image(image_path);
        std::cout << "Loaded input image: " << image_path << '\n';

        std::cout << "Loading model: " << model_path << '\n';
        clock_t start = clock();
        Graph graph = Loader::load(model_path.string());
        clock_t end = clock();

        std::cout << "Loading model finished after  " << double(end - start) /  CLOCKS_PER_SEC * 1000<< " ms \n";

        if (graph.inputs().size() != 1 || graph.outputs().size() != 1) {
            throw std::runtime_error("MNIST demo expects one input and one output");
        }

        const ValueId input_id = graph.inputs().front();
        const ValueId output_id = graph.outputs().front();
        const std::string output_name = graph.value(output_id).name();

        if (TensorDesc(image) != graph.desc(input_id)) {
            throw std::runtime_error("MNIST image shape does not match model input");
        }

        start = clock();

        Executor executor(graph);
        executor.set_input(input_id, std::move(image));

        executor.run();

        end = clock();

        std::cout << "execute model finished after  " << double(end - start) /  CLOCKS_PER_SEC * 1000 << " ms \n";


        const Tensor& logits = executor.get_output(output_name);
        const auto best = std::max_element(
            logits.raw_data(), logits.raw_data() + logits.numel());
        const std::size_t predicted =
            static_cast<std::size_t>(best - logits.raw_data());

        std::cout << "Output: " << output_name << '\n';
        print(logits);
        std::cout << "Predicted digit: " << predicted
                  << " (expected label: 7)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Inference failed: " << error.what() << '\n';
        return 1;
    }
}
