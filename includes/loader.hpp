#ifndef LOADER_HPP
#define LOADER_HPP

#include "graph.hpp"
#include <filesystem>

class Loader{
    public:
        static Graph load(const std::string path);
};

Tensor load_image(const std::filesystem::path& path);

#endif