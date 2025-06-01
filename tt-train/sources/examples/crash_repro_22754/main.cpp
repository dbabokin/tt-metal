// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <core/ttnn_all_includes.hpp>
#include <iostream>

template <typename T>
void generate_float_data(std::vector<T> &data, std::vector<uint32_t> const &shape) {
    size_t size = std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<size_t>());
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    data.resize(size);
    for (auto &val : data) {
        val = T(dis(gen));
    }
}

int main() {
    auto &system_mesh = tt::tt_metal::distributed::SystemMesh::instance();
    auto device_map = tt::tt_metal::distributed::MeshDevice::create_unit_meshes({0}, 24576, 0);
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> device = device_map.at(0);
    device->enable_program_cache();

    for (int i = 0; i < 10000000; i++) {
        std::cout << "\rIteration " << i;
        std::vector<uint32_t> shape = {32, 32};
        std::vector<bfloat16> datA, datMin, datMax;
        generate_float_data(datA, shape);
        generate_float_data(datMin, shape);
        generate_float_data(datMax, shape);

        tt::tt_metal::PageConfig page_config(tt::tt_metal::Layout::TILE);
        tt::tt_metal::MemoryConfig memory_config;

        auto tens_layout = tt::tt_metal::TensorLayout(tt::tt_metal::DataType::BFLOAT16, page_config, memory_config);
        auto tens_spec = ttnn::TensorSpec(ttnn::Shape(shape), tens_layout);

        auto cpu_tensor = tt::tt_metal::Tensor::from_vector<bfloat16>(datA, tens_spec);
        tt::tt_metal::Tensor tensor = cpu_tensor.to_device(device.get());

        auto cpu_tensor_min = tt::tt_metal::Tensor::from_vector<bfloat16>(datMin, tens_spec);
        auto cpu_tensor_max = tt::tt_metal::Tensor::from_vector<bfloat16>(datMax, tens_spec);
        tt::tt_metal::Tensor tensor_min = cpu_tensor_min.to_device(device.get());
        tt::tt_metal::Tensor tensor_max = cpu_tensor_max.to_device(device.get());

        auto result = ttnn::clamp(tensor, -1.0f, 1.0f);
        auto result2 = ttnn::clamp(tensor, tensor_min, tensor_max);

        tt::tt_metal::distributed::Synchronize(device.get(), std::nullopt, std::vector<tt::tt_metal::SubDeviceId>());
    }

    return 0;
}
