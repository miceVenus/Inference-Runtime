#ifndef LOADER_HPP
#define LOADER_HPP

#include "graph.hpp"
#include <filesystem>
#include <optional>
#include <vector>

class Loader{
    public:
        // Bind the batch dimension before graph shape inference.
        static Graph load(
            const std::string path,
            std::optional<long> batch_size = std::nullopt);
};

Tensor load_image(const std::filesystem::path& path);
Tensor load_image_batch(const std::vector<std::filesystem::path>& paths);

#endif
