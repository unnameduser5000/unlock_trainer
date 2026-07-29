#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/extension/training/module/bpfree_training_module.h>
#include <executorch/extension/training/module/training_module.h>
#include <executorch/runtime/platform/runtime.h>

namespace {

using executorch::aten::Tensor;
using executorch::extension::TensorPtr;
using executorch::extension::training::BPFreeTrainingModule;
using executorch::extension::training::TrainingModule;
using executorch::runtime::DataLoader;
using executorch::runtime::EValue;
using torch::executor::util::FileDataLoader;

constexpr executorch::aten::SizesType kHiddenSize = 2048;
constexpr char kMethod[] = "forward";

std::unique_ptr<DataLoader> open_model(const std::string& path) {
  auto result = FileDataLoader::from(path.c_str());
  if (!result.ok()) {
    throw std::runtime_error("Could not open PTE");
  }
  return std::make_unique<FileDataLoader>(std::move(result.get()));
}

struct Inputs {
  TensorPtr hidden;
  TensorPtr attention;
  TensorPtr positions;
  TensorPtr labels;
  std::vector<EValue> values;

  explicit Inputs(int64_t seq_len) {
    const auto sequence_size =
        static_cast<executorch::aten::SizesType>(seq_len);
    std::vector<float> hidden_values(seq_len * kHiddenSize);
    for (size_t index = 0; index < hidden_values.size(); ++index) {
      hidden_values[index] =
          static_cast<float>(static_cast<int>(index % 17) - 8) * 0.001f;
    }
    std::vector<float> attention_values(seq_len * seq_len, 0.0f);
    std::vector<int64_t> position_values(seq_len);
    std::vector<int64_t> label_values(seq_len, 1);
    for (int64_t index = 0; index < seq_len; ++index) {
      position_values[index] = index;
    }

    hidden = executorch::extension::make_tensor_ptr(
        {1, sequence_size, kHiddenSize}, std::move(hidden_values));
    attention = executorch::extension::make_tensor_ptr(
        {1, 1, sequence_size, sequence_size}, std::move(attention_values));
    positions = executorch::extension::make_tensor_ptr(
        {1, sequence_size}, std::move(position_values));
    labels = executorch::extension::make_tensor_ptr(
        {1, sequence_size}, std::move(label_values));
    values = {EValue(*hidden), EValue(*attention), EValue(*positions), EValue(*labels)};
  }
};

float max_gradient_difference(
    const std::map<std::string_view, Tensor>& split,
    const std::map<std::string_view, Tensor>& atomic) {
  if (split.size() != atomic.size()) {
    return std::numeric_limits<float>::infinity();
  }
  float maximum = 0.0f;
  for (const auto& [name, split_tensor] : split) {
    const auto found = atomic.find(name);
    if (found == atomic.end()) {
      return std::numeric_limits<float>::infinity();
    }
    const auto& atomic_tensor = found->second;
    if (split_tensor.scalar_type() != executorch::aten::ScalarType::Float ||
        atomic_tensor.scalar_type() != executorch::aten::ScalarType::Float ||
        split_tensor.numel() != atomic_tensor.numel()) {
      return std::numeric_limits<float>::infinity();
    }
    for (size_t index = 0; index < split_tensor.numel(); ++index) {
      maximum = std::max(
          maximum,
          std::abs(
              split_tensor.const_data_ptr<float>()[index] -
              atomic_tensor.const_data_ptr<float>()[index]));
    }
  }
  return maximum;
}

} // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: bpfree_module_harness <pte> <seq_len>\n";
    return 2;
  }
  try {
    executorch::runtime::runtime_init();
    const int64_t seq_len = std::stoll(argv[2]);
    Inputs inputs(seq_len);
    BPFreeTrainingModule split(open_model(argv[1]));
    TrainingModule atomic(open_model(argv[1]));

    // Match Android startup: checkpoint code asks for parameters first.
    auto parameters = split.named_parameters(kMethod);
    if (!parameters.ok() || parameters->empty()) {
      throw std::runtime_error("Could not load named parameters before forward");
    }

    auto boundary = split.forward_to_boundary(kMethod, inputs.values);
    if (!boundary.ok() || boundary->size() != 3 || !split.is_paused()) {
      const uint32_t boundary_error = boundary.ok()
          ? 0
          : static_cast<uint32_t>(boundary.error());
      std::cerr << "boundary_ok=" << boundary.ok()
                << " boundary_error=0x" << std::hex
                << boundary_error << std::dec
                << " outputs=" << (boundary.ok() ? boundary->size() : 0)
                << " paused=" << split.is_paused() << "\n";
      throw std::runtime_error("Forward boundary did not pause the Method");
    }
    if (boundary->at(1).toTensor().nbytes() == 0) {
      throw std::runtime_error("Boundary hidden tensor is empty");
    }
    if (split.forward_to_boundary(kMethod, inputs.values).ok()) {
      throw std::runtime_error("Paused module accepted a second forward");
    }

    auto resumed = split.resume_backward(kMethod);
    if (!resumed.ok() || split.is_paused()) {
      throw std::runtime_error("Backward did not finish the paused Method");
    }
    auto split_gradients = split.named_gradients(kMethod);
    if (!split_gradients.ok() || split_gradients->empty()) {
      throw std::runtime_error("Split execution produced no named gradients");
    }

    auto atomic_outputs = atomic.execute_forward_backward(kMethod, inputs.values);
    auto atomic_gradients = atomic.named_gradients(kMethod);
    if (!atomic_outputs.ok() || !atomic_gradients.ok()) {
      throw std::runtime_error("Atomic reference execution failed");
    }
    const float gradient_diff =
        max_gradient_difference(split_gradients.get(), atomic_gradients.get());

    // A second cycle checks reset_execution() and cached gradient aliases.
    auto second_boundary = split.forward_to_boundary(kMethod, inputs.values);
    auto second_resume = split.resume_backward(kMethod);
    if (!second_boundary.ok() || !second_resume.ok() || split.is_paused()) {
      throw std::runtime_error("Second split cycle failed");
    }

    const bool passed = gradient_diff <= 1.0e-5f;
    std::cout << "{\"parameters\":" << parameters->size()
              << ",\"gradients\":" << split_gradients->size()
              << ",\"hidden_bytes\":" << boundary->at(1).toTensor().nbytes()
              << ",\"belief_bytes\":" << boundary->at(2).toTensor().nbytes()
              << ",\"gradient_max_abs_diff\":" << gradient_diff
              << ",\"second_cycle\":true,\"passed\":"
              << (passed ? "true" : "false") << "}\n";
    return passed ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "bpfree_module_harness: " << error.what() << "\n";
    return 1;
  }
}
